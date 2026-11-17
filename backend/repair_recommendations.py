"""
repair_recommendations.py
-------------------------
Intelligent Repair Recommendation System for AutoGuard AI.

Exclusively supports the three trained ML classification categories:
    1. healthy
    2. misfire
    3. engine_knock

Respects the project's calibrated confidence and window-disagreement thresholds
(models/thresholds.json: abstain_threshold=0.75, disagree_threshold=0.20) and
never treats acoustic ML confidence as proof of a mechanical fault or proof that
an entire vehicle is 100% fault-free.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THRESHOLDS_PATH = os.path.join(_ROOT, "models", "thresholds.json")

# Strictly the three supported ML classification categories
SUPPORTED_CLASSES: tuple[str, ...] = ("healthy", "misfire", "engine_knock")

# Default confidence thresholds (calibrated on validation split in models/thresholds.json)
DEFAULT_HIGH_CONFIDENCE_THRESHOLD: float = 0.75
DEFAULT_MODERATE_CONFIDENCE_THRESHOLD: float = 0.55
DEFAULT_DISAGREE_THRESHOLD: float = 0.20


def load_confidence_thresholds() -> dict[str, float]:
    """Load calibrated confidence and disagreement thresholds from models/thresholds.json."""
    high_t = DEFAULT_HIGH_CONFIDENCE_THRESHOLD
    disagree_t = DEFAULT_DISAGREE_THRESHOLD
    if os.path.exists(_THRESHOLDS_PATH):
        try:
            with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            high_t = float(data.get("abstain_threshold", high_t))
            disagree_t = float(data.get("disagree_threshold", disagree_t))
        except Exception:
            pass
    return {
        "high_confidence_threshold": high_t,
        "moderate_confidence_threshold": DEFAULT_MODERATE_CONFIDENCE_THRESHOLD,
        "disagree_threshold": disagree_t,
    }


# Structured knowledge base exclusively for the three supported classes
REPAIR_RECOMMENDATIONS_DATA: dict[str, dict[str, Any]] = {
    "healthy": {
        "predicted_class": "healthy",
        "display_label": "Healthy",
        "diagnosis_overview": [
            "Engine sound appears consistent with the healthy class.",
            "No supported fault pattern was identified.",
            "Continue routine vehicle maintenance.",
            "Monitor unusual sounds if they develop.",
        ],
        "possible_causes": [],
        "recommended_inspections": [
            "Continue routine scheduled inspections according to the vehicle service manual.",
            "Monitor engine sound and dashboard indicators if new symptoms develop.",
        ],
        "possible_repairs": [],
        "safety_warning": (
            "Audio analysis covers only acoustic patterns for supported categories "
            "(healthy, misfire, engine_knock) and does not guarantee that the entire "
            "vehicle is 100% free of mechanical or electronic issues."
        ),
        "maintenance_advice": [
            "Engine sound appears consistent with the healthy class.",
            "No supported fault pattern was identified.",
            "Continue routine vehicle maintenance.",
            "Monitor unusual sounds if they develop.",
        ],
    },
    "misfire": {
        "predicted_class": "misfire",
        "display_label": "Misfire",
        "diagnosis_overview": [
            "Acoustic pattern is consistent with the Engine Misfire classification category.",
            "Audio screening indicates irregular combustion cadence; physical inspection is required to confirm the underlying cause.",
        ],
        "possible_causes": [
            "Worn or faulty spark plugs.",
            "Defective ignition coil.",
            "Fuel injector problems.",
            "Fuel delivery issues.",
        ],
        "recommended_inspections": [
            "Inspect spark plugs.",
            "Check ignition coils.",
            "Inspect fuel injectors.",
            "Check for engine warning lights and diagnostic trouble codes.",
        ],
        "possible_repairs": [
            "Clean or replace spark plugs if necessary.",
            "Replace a faulty ignition coil after confirmation.",
            "Clean or repair fuel injectors if testing confirms a problem.",
            "Repair fuel delivery issues if identified.",
        ],
        "safety_warning": (
            "If the engine is shaking severely, losing power, or the check-engine "
            "light is flashing, stop driving and arrange professional inspection."
        ),
        "maintenance_advice": [
            "Follow manufacturer spark plug and ignition service intervals.",
            "Use clean, manufacturer-recommended fuel and replace fuel filters on schedule.",
            "Address check-engine warning lights promptly before catalytic converter damage occurs.",
        ],
    },
    "engine_knock": {
        "predicted_class": "engine_knock",
        "display_label": "Engine Knocking",
        "diagnosis_overview": [
            "Acoustic pattern is consistent with the Engine Knocking classification category.",
            "Audio screening indicates metallic knocking or pinging tendencies; a professional mechanical check is needed to verify the source.",
        ],
        "possible_causes": [
            "Low engine oil level.",
            "Incorrect fuel quality or octane rating.",
            "Incorrect ignition timing.",
            "Internal engine component wear.",
            "Combustion-related problems.",
        ],
        "recommended_inspections": [
            "Check engine oil level and condition.",
            "Verify fuel quality and manufacturer's requirements.",
            "Inspect ignition and combustion systems.",
            "Have internal engine components inspected if knocking persists.",
        ],
        "possible_repairs": [
            "Top up or replace engine oil using the correct specification if required.",
            "Address fuel quality issues.",
            "Correct ignition-related problems if confirmed.",
            "Repair or replace damaged internal components if diagnosed by a mechanic.",
        ],
        "safety_warning": (
            "Loud or persistent knocking can indicate serious engine damage. "
            "Stop driving and arrange professional inspection."
        ),
        "maintenance_advice": [
            "Check engine oil level regularly and adhere to recommended oil change intervals.",
            "Always refuel with the octane rating specified in the owner's manual.",
            "Avoid heavy engine load or high RPM operation until knocking sounds are inspected.",
        ],
    },
}


def get_repair_recommendations(
    predicted_class: str,
    confidence: float,
    *,
    model_prediction: Optional[str] = None,
    uncertain: bool = False,
    uncertain_reason: Optional[str] = None,
    disagree_std: float = 0.0,
) -> dict[str, Any]:
    """
    Build a structured Repair Recommendation payload for a model prediction,
    respecting the calibrated confidence and uncertainty thresholds.

    Parameters
    ----------
    predicted_class : str
        Predicted label from the endpoint ("healthy", "misfire", "engine_knock", or "uncertain").
    confidence : float
        Top-class probability in [0.0, 1.0].
    model_prediction : Optional[str]
        Underlying top-scoring class before abstention (if predicted_class == "uncertain").
    uncertain : bool
        Whether the model flagged the prediction as uncertain.
    uncertain_reason : Optional[str]
        Explanation of why the prediction was marked uncertain.
    disagree_std : float
        Standard deviation of top-class confidence across audio windows.

    Returns
    -------
    dict[str, Any]
        Contains:
        - predicted_class
        - display_label
        - confidence
        - confidence_tier ("high" | "moderate" | "low" | "unsupported")
        - uncertainty_warning (Optional[str])
        - diagnosis_overview (list[str])
        - possible_causes (list[str])
        - recommended_inspections (list[str])
        - possible_repairs (list[str])
        - safety_warning (str)
        - maintenance_advice (list[str])
        - disclaimer (str)
    """
    thresholds = load_confidence_thresholds()
    high_t = thresholds["high_confidence_threshold"]
    mod_t = thresholds["moderate_confidence_threshold"]
    disagree_t = thresholds["disagree_threshold"]

    conf = round(float(max(0.0, min(1.0, confidence))), 4)

    # Resolve candidate class from either predicted_class or model_prediction
    raw_class = (
        model_prediction
        if predicted_class == "uncertain" and model_prediction in SUPPORTED_CLASSES
        else predicted_class
    )

    disclaimer = (
        "ML acoustic confidence is a screening indicator and is never proof of a "
        "specific mechanical fault. Always confirm diagnoses with a qualified mechanic."
    )

    # Handle unsupported category or unrecognized label gracefully
    if raw_class not in SUPPORTED_CLASSES:
        return {
            "predicted_class": "unsupported",
            "display_label": "Unsupported Audio",
            "confidence": conf,
            "confidence_tier": "unsupported",
            "uncertainty_warning": (
                "Unsupported audio or unrecognized classification category. AutoGuard AI "
                "exclusively supports healthy, misfire, and engine_knock engine recordings."
            ),
            "diagnosis_overview": [
                "The uploaded audio does not match a supported classification category.",
                "AutoGuard AI only evaluates healthy, misfire, and engine_knock patterns.",
            ],
            "possible_causes": [],
            "recommended_inspections": [
                "Verify that the recording captures a running vehicle engine in WAV or MP3 format.",
                "Re-record 10–15 seconds of steady engine idle close to the hood.",
            ],
            "possible_repairs": [],
            "safety_warning": (
                "If your vehicle exhibits warning lights, rough running, or unusual noises, "
                "arrange a professional mechanical inspection."
            ),
            "maintenance_advice": [
                "Continue routine vehicle maintenance according to manufacturer specifications."
            ],
            "disclaimer": disclaimer,
        }

    base = REPAIR_RECOMMENDATIONS_DATA[raw_class]
    high_window_disagreement = float(disagree_std) > disagree_t

    # 1. LOW CONFIDENCE (confidence < 0.55 OR severe window disagreement)
    # Rule: "Low confidence: Do not provide a definitive fault-specific diagnosis."
    if conf < mod_t or high_window_disagreement:
        reason_detail = (
            f" ({uncertain_reason})"
            if uncertain_reason
            else f" (confidence {conf * 100:.1f}% is below the {mod_t * 100:.0f}% minimum threshold)"
        )
        return {
            "predicted_class": raw_class,
            "display_label": "Inconclusive / Low Confidence",
            "confidence": conf,
            "confidence_tier": "low",
            "uncertainty_warning": (
                f"Low confidence reading{reason_detail}. Because the acoustic signal is ambiguous "
                "or inconsistent across time windows, a definitive fault-specific diagnosis cannot "
                "be provided from this recording."
            ),
            "diagnosis_overview": [
                "Audio confidence is too low to provide a definitive fault-specific diagnosis.",
                "Background noise, microphone distance, or fluctuating engine speed may have obscured the sound pattern.",
                "Re-record a clean 10–15 second clip at steady engine idle, or have a technician inspect the vehicle directly.",
            ],
            "possible_causes": [],
            "recommended_inspections": [
                "Re-record engine audio 1–2 feet from the open hood in a quiet setting.",
                "Check the dashboard for any active check-engine or oil warning lights.",
                "Arrange a professional inspection if you notice roughness, power loss, or metallic noise.",
            ],
            "possible_repairs": [],
            "safety_warning": (
                base["safety_warning"]
                if raw_class in ("misfire", "engine_knock")
                else (
                    "Low-confidence audio cannot rule out mechanical issues. If the vehicle "
                    "shakes, loses power, or makes loud knocking sounds, stop driving and "
                    "arrange professional inspection."
                )
            ),
            "maintenance_advice": [
                "Continue routine vehicle maintenance.",
                "Monitor unusual sounds if they develop.",
            ],
            "disclaimer": disclaimer,
        }

    # 2. MODERATE CONFIDENCE (0.55 <= confidence < abstain_threshold [0.75] or uncertain flag)
    # Rule: "Moderate confidence: Display recommendations with an uncertainty warning."
    if conf < high_t or uncertain:
        return {
            "predicted_class": base["predicted_class"],
            "display_label": base["display_label"],
            "confidence": conf,
            "confidence_tier": "moderate",
            "uncertainty_warning": (
                f"Moderate confidence ({conf * 100:.1f}% — below the {high_t * 100:.0f}% high-confidence "
                "threshold). The recommendations below are preliminary possibilities based on acoustic "
                "similarity and must be verified through physical inspection before any repair."
            ),
            "diagnosis_overview": list(base["diagnosis_overview"]),
            "possible_causes": list(base["possible_causes"]),
            "recommended_inspections": list(base["recommended_inspections"]),
            "possible_repairs": list(base["possible_repairs"]),
            "safety_warning": base["safety_warning"],
            "maintenance_advice": list(base["maintenance_advice"]),
            "disclaimer": disclaimer,
        }

    # 3. HIGH CONFIDENCE (confidence >= abstain_threshold [0.75] and low window disagreement)
    # Rule: "High confidence: Display relevant recommendations."
    return {
        "predicted_class": base["predicted_class"],
        "display_label": base["display_label"],
        "confidence": conf,
        "confidence_tier": "high",
        "uncertainty_warning": None,
        "diagnosis_overview": list(base["diagnosis_overview"]),
        "possible_causes": list(base["possible_causes"]),
        "recommended_inspections": list(base["recommended_inspections"]),
        "possible_repairs": list(base["possible_repairs"]),
        "safety_warning": base["safety_warning"],
        "maintenance_advice": list(base["maintenance_advice"]),
        "disclaimer": disclaimer,
    }
