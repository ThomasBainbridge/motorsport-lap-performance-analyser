from __future__ import annotations

import numpy as np
import pandas as pd


LOSS_SIGN_FEATURES = {
    "brake_start_delta_m": 1.0,
    "entry_speed_delta_kph": -1.0,
    "brake_fraction_delta": 1.0,
    "min_speed_delta_kph": -1.0,
    "apex_delta_m": 1.0,
    "throttle_pickup_delta_m": 1.0,
    "exit_speed_delta_kph": -1.0,
    "mean_throttle_delta_pct": -1.0,
    "apex_to_exit_gain_delta_kph": -1.0,
}

PHASE_GROUPS = {
    "Braking": ["brake_start_delta_m", "entry_speed_delta_kph", "brake_fraction_delta"],
    "Minimum-speed": ["min_speed_delta_kph", "apex_delta_m"],
    "Traction/exit": ["throttle_pickup_delta_m", "exit_speed_delta_kph", "mean_throttle_delta_pct", "apex_to_exit_gain_delta_kph"],
}

FEATURE_PHRASES = {
    "brake_start_delta_m": ("later braking", "earlier braking"),
    "entry_speed_delta_kph": ("lower entry speed", "higher entry speed"),
    "brake_fraction_delta": ("more time on the brake", "less time on the brake"),
    "min_speed_delta_kph": ("lower minimum speed", "higher minimum speed"),
    "apex_delta_m": ("later apex timing", "earlier apex timing"),
    "throttle_pickup_delta_m": ("later throttle pickup", "earlier throttle pickup"),
    "exit_speed_delta_kph": ("weaker exit speed", "stronger exit speed"),
    "mean_throttle_delta_pct": ("lower average throttle", "higher average throttle"),
    "apex_to_exit_gain_delta_kph": ("weaker post-apex acceleration", "stronger post-apex acceleration"),
}

THRESHOLDS = {
    "brake_start_delta_m": 5.0,
    "entry_speed_delta_kph": 2.0,
    "brake_fraction_delta": 0.03,
    "min_speed_delta_kph": 2.0,
    "apex_delta_m": 5.0,
    "throttle_pickup_delta_m": 8.0,
    "exit_speed_delta_kph": 2.0,
    "mean_throttle_delta_pct": 2.5,
    "apex_to_exit_gain_delta_kph": 2.0,
}


def _segment_descriptor(row) -> str:
    label = getattr(row, "SegmentLabel", None)
    if isinstance(label, str) and label.strip():
        return label
    return f"S{int(row.SegmentId)}"



def compute_segment_contributions(segment_features: pd.DataFrame) -> pd.DataFrame:
    if segment_features.empty:
        return pd.DataFrame(columns=["SegmentId", "SegmentLabel", "Phase", "ContributionScore"])

    rows: list[dict[str, float | str]] = []
    for row in segment_features.itertuples(index=False):
        row_dict = row._asdict()
        for phase_name, features in PHASE_GROUPS.items():
            score = 0.0
            for feature in features:
                value = float(row_dict.get(feature, 0.0))
                score += LOSS_SIGN_FEATURES[feature] * value
            rows.append(
                {
                    "SegmentId": int(row_dict["SegmentId"]),
                    "SegmentLabel": row_dict.get("SegmentLabel", f"S{int(row_dict['SegmentId'])}"),
                    "Phase": phase_name,
                    "ContributionScore": float(score),
                }
            )
    contributions = pd.DataFrame(rows)
    max_abs = contributions["ContributionScore"].abs().max()
    if pd.notna(max_abs) and max_abs > 0:
        contributions["NormalizedContribution"] = contributions["ContributionScore"] / max_abs
    else:
        contributions["NormalizedContribution"] = 0.0
    return contributions



def _feature_phrase(feature_name: str, value: float, time_loss_s: float) -> str | None:
    threshold = THRESHOLDS.get(feature_name, 0.0)
    if abs(value) < threshold:
        return None

    loss_direction = LOSS_SIGN_FEATURES[feature_name] * value
    loss_phrase, gain_phrase = FEATURE_PHRASES[feature_name]

    if time_loss_s >= 0:
        if loss_direction <= 0:
            return None
        return loss_phrase

    if loss_direction >= 0:
        return None
    return gain_phrase



def rank_segments(segment_features: pd.DataFrame) -> pd.DataFrame:
    if segment_features.empty:
        return segment_features.copy()

    ranked = segment_features.copy()
    if "SegmentLabel" not in ranked.columns:
        ranked["SegmentLabel"] = ranked["SegmentId"].map(lambda x: f"S{int(x)}")
    ranked["abs_time_loss_s"] = ranked["time_loss_s"].abs()
    ranked = ranked.sort_values(["abs_time_loss_s", "SegmentId"], ascending=[False, True]).reset_index(drop=True)

    narratives: list[str] = []
    dominant_phases: list[str] = []
    for row in ranked.itertuples(index=False):
        row_dict = row._asdict()
        phrase_candidates: list[tuple[float, str]] = []
        phase_scores: dict[str, float] = {}
        for phase_name, features in PHASE_GROUPS.items():
            phase_score = 0.0
            for feature in features:
                value = float(row_dict.get(feature, 0.0))
                phase_score += LOSS_SIGN_FEATURES[feature] * value
                phrase = _feature_phrase(feature, value, row.time_loss_s)
                if phrase is not None:
                    phrase_candidates.append((abs(value), phrase))
            phase_scores[phase_name] = abs(phase_score)

        if not phrase_candidates:
            phrase_candidates.append((0.0, "small distributed effects"))
        ordered_phases = sorted(phase_scores.items(), key=lambda item: item[1], reverse=True)
        dominant_phase = ordered_phases[0][0] if ordered_phases else "Mixed"
        dominant_phases.append(dominant_phase)

        sign = "lost" if row.time_loss_s >= 0 else "gained"
        label = _segment_descriptor(row)
        unique_phrases = []
        for _, phrase in sorted(phrase_candidates, key=lambda item: item[0], reverse=True):
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
        narratives.append(
            f"{label}: {sign} {abs(row.time_loss_s):.3f} s, driven mainly by {dominant_phase.lower()} effects such as "
            + ", ".join(unique_phrases[:4])
            + "."
        )

    ranked["DominantPhase"] = dominant_phases
    ranked["Narrative"] = narratives
    return ranked



def overall_summary(aligned_df: pd.DataFrame) -> dict[str, float]:
    total_delta = float(aligned_df["DeltaSeconds"].iloc[-1] - aligned_df["DeltaSeconds"].iloc[0])
    max_loss = float(np.max(aligned_df["DeltaSeconds"]))
    min_gain = float(np.min(aligned_df["DeltaSeconds"]))
    return {
        "total_delta_s": total_delta,
        "max_cumulative_loss_s": max_loss,
        "max_cumulative_gain_s": min_gain,
    }
