import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { ContactShadows, Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import {
  Activity,
  AlertTriangle,
  Box,
  Camera,
  CheckCircle2,
  Cpu,
  Eye,
  Flame,
  Info,
  Layers,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Wrench,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { ConfidenceTier, EngineCondition } from "../types/diagnostic";
import { predictionToLabel } from "../services/api";
import { Badge, Button } from "./ui";

export type RenderStyle = "xray" | "metallic" | "blueprint";
export type CameraPresetId = "hero" | "cranktrain" | "cylinders" | "valvetrain" | "combustion" | "ignition";

export interface EngineComponentSpec {
  id: string;
  name: string;
  category: "ignition_fuel" | "combustion_rotating" | "valvetrain_timing" | "structure_lubrication";
  categoryLabel: string;
  nodePrefixes: string[];
  partCount: number;
  materialSpec: string;
  functionSummary: string;
  acousticSignature: string;
  misfireRelevance: string | null;
  knockRelevance: string | null;
  inspectionSteps: string[];
  explodeOffset: [number, number, number];
  cameraTarget: [number, number, number];
  cameraPos: [number, number, number];
}

export const ENGINE_COMPONENTS: EngineComponentSpec[] = [
  {
    id: "spark_plugs",
    name: "Spark Plugs (Cyl 1–4)",
    category: "ignition_fuel",
    categoryLabel: "Ignition & Fuel",
    nodePrefixes: ["Spark_Plug_"],
    partCount: 4,
    materialSpec: "Iridium Center Electrode / Alumina Ceramic Insulator",
    functionSummary:
      "Delivers the high-voltage ignition arc across the electrode gap to initiate air-fuel combustion at top dead center.",
    acousticSignature:
      "Fouled, bridged, or worn electrode gaps cause incomplete or skipped cylinder firing, producing an irregular exhaust cadence.",
    misfireRelevance:
      "Possible Inspection Area for Misfire: Worn or faulty spark plugs are a primary cause of uneven cylinder firing.",
    knockRelevance:
      "Incorrect heat-range plugs or carbon deposits on the insulator tip can create hot spots that trigger pre-ignition.",
    inspectionSteps: [
      "Inspect spark plugs for electrode wear, carbon fouling, oil deposits, or cracked ceramic insulators.",
      "Measure electrode gap against manufacturer specification using a feeler gauge.",
      "Clean or replace spark plugs if testing confirms wear or fouling.",
    ],
    explodeOffset: [0, 0.78, 0],
    cameraTarget: [0, 0.95, 0],
    cameraPos: [1.6, 1.9, 2.5],
  },
  {
    id: "ignition_coils",
    name: "Ignition Coils (COP 1–4)",
    category: "ignition_fuel",
    categoryLabel: "Ignition & Fuel",
    nodePrefixes: ["Ignition_Coil_"],
    partCount: 4,
    materialSpec: "Epoxy-Potted Copper Windings / Silicone Boots",
    functionSummary:
      "Transforms 12V battery potential into 25,000–45,000V ignition pulses directly above each cylinder's spark plug.",
    acousticSignature:
      "A weak or shorted coil winding drops spark energy under load, causing rhythmic stumbles and unburned fuel pulses.",
    misfireRelevance:
      "Possible Inspection Area for Misfire: A defective ignition coil can interrupt spark delivery and cause cylinder misfire.",
    knockRelevance: null,
    inspectionSteps: [
      "Check ignition coils for boot cracking, carbon tracking, or primary/secondary resistance out of spec.",
      "Check for engine warning lights and OBD-II diagnostic trouble codes (P0300–P0304).",
      "Replace a faulty ignition coil after confirmation.",
    ],
    explodeOffset: [0, 1.38, 0],
    cameraTarget: [0, 1.45, 0],
    cameraPos: [1.9, 2.3, 2.7],
  },
  {
    id: "fuel_injectors",
    name: "Fuel Injectors & Common Rail",
    category: "ignition_fuel",
    categoryLabel: "Ignition & Fuel",
    nodePrefixes: ["Fuel_Injector_", "Fuel_Rail"],
    partCount: 5,
    materialSpec: "Stainless Steel High-Pressure Rail / Solenoid Piezo Nozzles",
    functionSummary:
      "Atomizes pressurized fuel at precise millisecond pulse widths into the intake ports and combustion chambers.",
    acousticSignature:
      "Healthy injectors produce a crisp high-frequency solenoid tick; clogged or leaking nozzles cause lean/rich misfires.",
    misfireRelevance:
      "Possible Inspection Area for Misfire: Fuel injector problems or fuel delivery issues starve the cylinder of proper mixture.",
    knockRelevance:
      "A partially clogged injector can lean out air-fuel mixture in a cylinder, raising combustion temperature and knock risk.",
    inspectionSteps: [
      "Inspect fuel injectors for clogged spray nozzles, electrical connector faults, or seal leaks.",
      "Verify fuel rail pressure and check for fuel delivery issues.",
      "Clean or repair fuel injectors if testing confirms a problem.",
    ],
    explodeOffset: [0, 0.42, 0.82],
    cameraTarget: [0, 1.12, 0.55],
    cameraPos: [1.5, 1.7, 2.8],
  },
  {
    id: "combustion_chamber",
    name: "Combustion Chambers (Cyl 1–4)",
    category: "combustion_rotating",
    categoryLabel: "Combustion & Rotating",
    nodePrefixes: ["Combustion_Chamber_"],
    partCount: 4,
    materialSpec: "Pent-Roof Hemispherical Dome Volume",
    functionSummary:
      "Contains the high-pressure flame front where compressed air-fuel mixture ignites and drives the piston downward.",
    acousticSignature:
      "Abnormal end-gas auto-ignition creates sharp 5–7 kHz acoustic pressure oscillations (metallic pinging/knocking).",
    misfireRelevance:
      "Loss of combustion pressure during a misfire leaves the chamber cold and unlit during its scheduled power stroke.",
    knockRelevance:
      "Possible Inspection Area for Engine Knocking: Combustion-related problems, incorrect fuel octane, or advanced ignition timing cause detonation pressure spikes.",
    inspectionSteps: [
      "Verify fuel quality and manufacturer's octane rating requirements.",
      "Inspect ignition timing and combustion systems.",
      "Perform a cylinder compression and leak-down test if knocking or misfire persists.",
    ],
    explodeOffset: [0, 0.25, 0],
    cameraTarget: [0, 0.56, 0],
    cameraPos: [0.9, 1.1, 2.4],
  },
  {
    id: "pistons",
    name: "Forged Pistons & Rings (Cyl 1–4)",
    category: "combustion_rotating",
    categoryLabel: "Combustion & Rotating",
    nodePrefixes: ["Piston_"],
    partCount: 4,
    materialSpec: "4032-T6 Billet Aluminum / Moly-Faced Compression Rings",
    functionSummary:
      "Transfers combustion gas force into reciprocating mechanical motion while sealing the cylinder bore against blow-by.",
    acousticSignature:
      "Detonation shockwaves hammer the piston crown and ring lands, creating a distinct metallic knocking sound.",
    misfireRelevance: null,
    knockRelevance:
      "Possible Inspection Area for Engine Knocking: Pistons absorb abnormal detonation shockwaves and can exhibit piston slap or crown/ring land wear.",
    inspectionSteps: [
      "Have internal engine components inspected with a borescope through the spark plug wells if knocking persists.",
      "Listen for cold vs. warm piston slap and check cylinder compression balance.",
      "Repair or replace damaged internal components if diagnosed by a mechanic.",
    ],
    explodeOffset: [0, -0.18, 0],
    cameraTarget: [0, 0.05, 0],
    cameraPos: [1.4, 0.65, 2.6],
  },
  {
    id: "connecting_rods",
    name: "Connecting Rods & Bearings",
    category: "combustion_rotating",
    categoryLabel: "Combustion & Rotating",
    nodePrefixes: ["Connecting_Rod_"],
    partCount: 4,
    materialSpec: "4340 Forged Chromoly H-Beam / Tri-Metal Rod Bearings",
    functionSummary:
      "Links each reciprocating piston wrist pin to the rotating crankshaft journal under high tensile and compressive loads.",
    acousticSignature:
      "Worn rod bearings or oil starvation produce a deep, hollow metallic knock that increases with engine RPM.",
    misfireRelevance: null,
    knockRelevance:
      "Possible Inspection Area for Engine Knocking: Internal engine component wear (such as rod bearings) or low engine oil level can cause severe mechanical knocking.",
    inspectionSteps: [
      "Check engine oil level and condition immediately; inspect oil for metallic debris.",
      "Have internal engine components and rod bearing clearances inspected if knocking persists.",
      "Repair or replace damaged internal components if diagnosed by a mechanic.",
    ],
    explodeOffset: [0, -0.45, 0],
    cameraTarget: [0, -0.35, 0],
    cameraPos: [1.6, -0.1, 2.6],
  },
  {
    id: "crankshaft",
    name: "Counterweighted Crankshaft & Flywheel",
    category: "combustion_rotating",
    categoryLabel: "Combustion & Rotating",
    nodePrefixes: ["Crankshaft"],
    partCount: 1,
    materialSpec: "Induction-Hardened Forged Steel / Balanced 180° Flat-Plane Throws",
    functionSummary:
      "Converts reciprocating forces from all four cylinders (1-3-4-2 firing order) into smooth rotational torque.",
    acousticSignature:
      "Main bearing wear or severe torsional damper imbalance generates low-frequency rumble and bottom-end knock.",
    misfireRelevance:
      "Crankshaft angular velocity sensor (CKP) detects misfires by measuring micro-decelerations between firing pulses.",
    knockRelevance:
      "Possible Inspection Area for Engine Knocking: Main journal bearings depend on proper oil pressure and film strength to prevent bottom-end knock.",
    inspectionSteps: [
      "Check engine oil level and specification; verify oil pressure at warm idle.",
      "Inspect crankshaft damper pulley and main bearings if bottom-end noise persists.",
    ],
    explodeOffset: [0, -0.78, 0],
    cameraTarget: [0, -0.95, 0],
    cameraPos: [2.1, -0.55, 2.9],
  },
  {
    id: "valvetrain",
    name: "DOHC Camshafts & 16-Valve Assembly",
    category: "valvetrain_timing",
    categoryLabel: "Valvetrain & Timing",
    nodePrefixes: ["Camshaft_Intake", "Camshaft_Exhaust", "Intake_Valves", "Exhaust_Valves"],
    partCount: 4,
    materialSpec: "Chilled Cast Iron Billet Cams / Heat-Treated Steel Poppet Valves",
    functionSummary:
      "Opens and closes intake and exhaust valves in exact synchronization with crankshaft rotation.",
    acousticSignature:
      "Excessive valve lash or low upper-head oil supply produces a rapid valvetrain ticking at half crankshaft speed.",
    misfireRelevance:
      "Improper valve sealing can reduce cylinder compression and contribute to rough idle.",
    knockRelevance:
      "Low oil level can starve upper cam followers and hydraulic lifters, producing valvetrain clatter.",
    inspectionSteps: [
      "Check engine oil level and viscosity to ensure proper upper-valvetrain lubrication.",
      "Inspect valve clearance and cam lobe condition during scheduled maintenance.",
    ],
    explodeOffset: [0, 0.58, 0],
    cameraTarget: [0, 1.22, 0],
    cameraPos: [1.8, 1.85, 2.5],
  },
  {
    id: "timing_belt",
    name: "Timing Drive Belt & Sprockets",
    category: "valvetrain_timing",
    categoryLabel: "Valvetrain & Timing",
    nodePrefixes: ["Timing_Drive_Belt"],
    partCount: 1,
    materialSpec: "HNBR Aramid-Reinforced Synchronous Belt / Steel Tensioner",
    functionSummary:
      "Synchronizes crankshaft rotation with the intake and exhaust camshafts at a 2:1 ratio.",
    acousticSignature:
      "Worn tensioner bearings or belt misalignment create rhythmic whirring or chirping at the front cover.",
    misfireRelevance: null,
    knockRelevance:
      "Incorrect cam/ignition timing synchronization can alter cylinder pressure and trigger pinging.",
    inspectionSteps: [
      "Inspect timing belt tension, tooth condition, and idler pulley bearings at manufacturer intervals.",
      "Verify cam-to-crank timing marks if ignition timing issues are suspected.",
    ],
    explodeOffset: [-0.88, 0, 0],
    cameraTarget: [-2.1, 0.3, 0],
    cameraPos: [-3.4, 0.9, 2.2],
  },
  {
    id: "oil_pan",
    name: "Lubrication Sump & Oil Pan",
    category: "structure_lubrication",
    categoryLabel: "Block & Lubrication",
    nodePrefixes: ["Oil_Pan"],
    partCount: 1,
    materialSpec: "Die-Cast Baffled Aluminum Sump",
    functionSummary:
      "Stores and cools engine oil supplying pressurized lubrication to crankshaft bearings, pistons, and camshafts.",
    acousticSignature:
      "Low oil level or degraded oil viscosity reduces hydrodynamic cushioning, amplifying metallic knocking sounds.",
    misfireRelevance: null,
    knockRelevance:
      "Possible Inspection Area for Engine Knocking: Low engine oil level or degraded oil specification is a primary cause of engine knocking.",
    inspectionSteps: [
      "Check engine oil level on the dipstick and inspect oil condition.",
      "Top up or replace engine oil using the correct specification if required.",
    ],
    explodeOffset: [0, -1.25, 0],
    cameraTarget: [0, -1.55, 0],
    cameraPos: [1.8, -1.1, 2.9],
  },
  {
    id: "engine_block",
    name: "Cylinder Block, Head & Exhaust Header",
    category: "structure_lubrication",
    categoryLabel: "Block & Lubrication",
    nodePrefixes: ["Engine_Block", "Cylinder_Head", "Valve_Cover", "Exhaust_Manifold"],
    partCount: 4,
    materialSpec: "A356-T6 Cast Aluminum Block / Cast Iron Cylinder Liners",
    functionSummary:
      "Forms the structural core housing the four cylinder bores, cooling jackets, cylinder head, and exhaust collector.",
    acousticSignature:
      "Transmits internal combustion and mechanical vibration resonances outward to the engine bay.",
    misfireRelevance: null,
    knockRelevance: null,
    inspectionSteps: [
      "Continue routine vehicle maintenance and coolant/oil checks.",
      "Monitor unusual sounds if they develop.",
    ],
    explodeOffset: [0, 0.28, -0.35],
    cameraTarget: [0, 0.3, 0],
    cameraPos: [2.8, 1.5, 3.6],
  },
];

export const CAMERA_PRESETS: Array<{
  id: CameraPresetId;
  num: number;
  label: string;
  position: [number, number, number];
  target: [number, number, number];
}> = [
  { id: "hero", num: 1, label: "Hero Cutaway", position: [3.1, 1.65, 3.55], target: [0, 0.15, 0] },
  { id: "cranktrain", num: 2, label: "Cranktrain", position: [2.2, -0.55, 3.1], target: [0, -0.55, 0] },
  { id: "cylinders", num: 3, label: "Cylinders 1–4", position: [0.25, 0.45, 3.45], target: [0, 0.2, 0] },
  { id: "valvetrain", num: 4, label: "DOHC Valvetrain", position: [2.0, 2.25, 2.55], target: [0, 1.25, 0] },
  { id: "combustion", num: 5, label: "Combustion", position: [1.15, 0.95, 2.35], target: [0, 0.55, 0] },
  { id: "ignition", num: 6, label: "Ignition & Fuel", position: [1.65, 2.1, 2.65], target: [0, 1.25, 0.25] },
];

export function isHighlightedForPrediction(
  compId: string,
  prediction: EngineCondition,
  confidenceTier: ConfidenceTier = "high"
): boolean {
  if (confidenceTier === "low" || confidenceTier === "unsupported") return false;
  if (prediction === "misfire") {
    return compId === "spark_plugs" || compId === "ignition_coils" || compId === "fuel_injectors";
  }
  if (prediction === "engine_knock") {
    return (
      compId === "combustion_chamber" ||
      compId === "pistons" ||
      compId === "connecting_rods" ||
      compId === "crankshaft" ||
      compId === "oil_pan"
    );
  }
  return false;
}

function findComponentByNodeName(nodeName: string): EngineComponentSpec | undefined {
  return ENGINE_COMPONENTS.find((c) => c.nodePrefixes.some((p) => nodeName.startsWith(p)));
}

function checkWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch {
    return false;
  }
}

/* ── Camera Controller inside R3F Canvas ───────────────────────────────────── */
function CameraRig({
  preset,
  selectedSpec,
  controlsRef,
}: {
  preset: CameraPresetId;
  selectedSpec: EngineComponentSpec | null;
  controlsRef: React.RefObject<any>;
}) {
  const { camera } = useThree();
  const desiredPos = useRef(new THREE.Vector3(3.1, 1.65, 3.55));
  const desiredTarget = useRef(new THREE.Vector3(0, 0.15, 0));
  const animating = useRef(false);

  useEffect(() => {
    const found = CAMERA_PRESETS.find((p) => p.id === preset) || CAMERA_PRESETS[0];
    desiredPos.current.set(...found.position);
    desiredTarget.current.set(...found.target);
    animating.current = true;
  }, [preset]);

  useEffect(() => {
    if (!selectedSpec) return;
    desiredPos.current.set(...selectedSpec.cameraPos);
    desiredTarget.current.set(...selectedSpec.cameraTarget);
    animating.current = true;
  }, [selectedSpec]);

  useFrame((_, delta) => {
    if (!animating.current) return;
    const step = Math.min(1, delta * 4.5);
    camera.position.lerp(desiredPos.current, step);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(desiredTarget.current, step);
      controlsRef.current.update();
    }
    if (camera.position.distanceTo(desiredPos.current) < 0.04) {
      animating.current = false;
    }
  });

  return null;
}

/* ── GLB 3D Engine Scene Graph & Kinematic Animator ─────────────────────────── */
function EngineGlbScene({
  prediction,
  confidenceTier,
  renderStyle,
  explodeAmount,
  rpm,
  playing,
  selectedId,
  hoveredId,
  onSelectComponent,
  onHoverComponent,
  onCrankAngleUpdate,
}: {
  prediction: EngineCondition;
  confidenceTier: ConfidenceTier;
  renderStyle: RenderStyle;
  explodeAmount: number;
  rpm: number;
  playing: boolean;
  selectedId: string | null;
  hoveredId: string | null;
  onSelectComponent: (id: string) => void;
  onHoverComponent: (id: string | null) => void;
  onCrankAngleUpdate: (deg: number) => void;
}) {
  const gltf = useGLTF("/models/autoguard_i4_engine.glb");
  const crankAngleRef = useRef(0);
  const frameCounterRef = useRef(0);

  // Clone scene & materials so we can mutate positions/materials per frame cleanly
  const { clonedScene, nodeMap, baseTranslations } = useMemo(() => {
    const clone = gltf.scene.clone(true);
    const map: Record<string, THREE.Object3D> = {};
    const bases: Record<string, THREE.Vector3> = {};

    clone.traverse((obj) => {
      if (obj.name) {
        map[obj.name] = obj;
        bases[obj.name] = obj.position.clone();
      }
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        if (mesh.material) {
          mesh.material = (mesh.material as THREE.MeshStandardMaterial).clone();
        }
      }
    });

    return { clonedScene: clone, nodeMap: map, baseTranslations: bases };
  }, [gltf.scene]);

  // Dynamic materials / highlights update
  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime();
    if (playing) {
      // Convert RPM to radians/sec (scaled slightly for visual clarity at 60fps)
      const radPerSec = (rpm / 60) * Math.PI * 2 * 0.18;
      crankAngleRef.current = (crankAngleRef.current + delta * radPerSec) % (Math.PI * 4);
      frameCounterRef.current += 1;
      if (frameCounterRef.current % 4 === 0) {
        onCrankAngleUpdate(Math.round((crankAngleRef.current * 180) / Math.PI) % 720);
      }
    }

    const theta = crankAngleRef.current;
    const pulse = 0.5 + 0.5 * Math.sin(t * 5.0);
    const phases = [0, Math.PI, Math.PI, 0];
    // 4-stroke firing offsets for 1-3-4-2 order over 720 deg (4*pi rad)
    const firingCenters = [0, Math.PI * 3, Math.PI * 1, Math.PI * 2];

    // 1. Animate Crankshaft & Camshafts
    const crank = nodeMap["Crankshaft"];
    if (crank) {
      crank.rotation.x = theta;
    }
    const camIn = nodeMap["Camshaft_Intake"];
    const camEx = nodeMap["Camshaft_Exhaust"];
    if (camIn) camIn.rotation.x = theta * 0.5;
    if (camEx) camEx.rotation.x = theta * 0.5;

    // 2. Animate Pistons, Connecting Rods & Combustion Chambers
    for (let i = 0; i < 4; i++) {
      const cNum = i + 1;
      const ph = theta + phases[i];
      const strokeY = Math.cos(ph) * 0.26;
      const rodAngle = -Math.sin(ph) * 0.18;

      const piston = nodeMap[`Piston_${cNum}`];
      const rod = nodeMap[`Connecting_Rod_${cNum}`];
      const chamber = nodeMap[`Combustion_Chamber_${cNum}`] as THREE.Mesh | undefined;

      const pistonSpec = ENGINE_COMPONENTS.find((c) => c.id === "pistons")!;
      const rodSpec = ENGINE_COMPONENTS.find((c) => c.id === "connecting_rods")!;

      if (piston) {
        const base = baseTranslations[`Piston_${cNum}`];
        piston.position.set(
          base.x + pistonSpec.explodeOffset[0] * explodeAmount,
          0.05 + strokeY + pistonSpec.explodeOffset[1] * explodeAmount,
          base.z + pistonSpec.explodeOffset[2] * explodeAmount
        );
      }

      if (rod) {
        const base = baseTranslations[`Connecting_Rod_${cNum}`];
        rod.position.set(
          base.x + rodSpec.explodeOffset[0] * explodeAmount,
          0.05 + strokeY + rodSpec.explodeOffset[1] * explodeAmount,
          base.z + rodSpec.explodeOffset[2] * explodeAmount
        );
        rod.rotation.x = rodAngle;
      }

      // Combustion chamber volumetric glow in 1-3-4-2 firing order
      if (chamber && chamber.material) {
        const mat = chamber.material as THREE.MeshStandardMaterial;
        const dist = Math.abs(((theta - firingCenters[i] + Math.PI * 6) % (Math.PI * 4)) - Math.PI * 2);
        const isFiring = dist < 0.65;
        const fireStrength = isFiring ? Math.max(0, 1 - dist / 0.65) : 0;

        if (prediction === "misfire" && cNum === 3) {
          // Cylinder 3 misfire: no flame / cold dark red flicker
          mat.color.set("#EF4444");
          mat.emissive.set("#7F1D1D");
          mat.emissiveIntensity = 0.25 * pulse;
          mat.opacity = 0.28;
        } else if (prediction === "engine_knock" && (cNum === 2 || cNum === 3)) {
          // Detonation knock: intense golden-amber shock pulse
          const knockSpike = isFiring ? 1.8 + Math.sin(t * 45) * 0.6 : 0.45 * pulse;
          mat.color.set("#F59E0B");
          mat.emissive.set("#F59E0B");
          mat.emissiveIntensity = knockSpike;
          mat.opacity = isFiring ? 0.82 : 0.42;
        } else {
          // Healthy combustion pulse
          mat.color.set("#38BDF8");
          mat.emissive.set(isFiring ? "#F59E0B" : "#0284C7");
          mat.emissiveIntensity = isFiring ? fireStrength * 1.4 : 0.15;
          mat.opacity = isFiring ? 0.68 : 0.2;
        }
      }
    }

    // 3. Animate Valves
    const inValves = nodeMap["Intake_Valves"];
    const exValves = nodeMap["Exhaust_Valves"];
    const valveExplode = ENGINE_COMPONENTS.find((c) => c.id === "valvetrain")!.explodeOffset;
    if (inValves) {
      const b = baseTranslations["Intake_Valves"];
      const lift = Math.max(0, Math.sin(theta * 0.5)) * 0.05;
      inValves.position.set(
        b.x + valveExplode[0] * explodeAmount,
        b.y - lift + valveExplode[1] * explodeAmount,
        b.z + valveExplode[2] * explodeAmount
      );
    }
    if (exValves) {
      const b = baseTranslations["Exhaust_Valves"];
      const lift = Math.max(0, Math.cos(theta * 0.5)) * 0.05;
      exValves.position.set(
        b.x + valveExplode[0] * explodeAmount,
        b.y - lift + valveExplode[1] * explodeAmount,
        b.z + valveExplode[2] * explodeAmount
      );
    }

    // 4. Apply Explosion Vectors & Material Styling to all Nodes
    Object.entries(nodeMap).forEach(([nodeName, obj]) => {
      const spec = findComponentByNodeName(nodeName);
      if (!spec) return;

      // Position explosion for non-kinematic nodes
      if (
        !nodeName.startsWith("Piston_") &&
        !nodeName.startsWith("Connecting_Rod_") &&
        nodeName !== "Intake_Valves" &&
        nodeName !== "Exhaust_Valves"
      ) {
        const b = baseTranslations[nodeName];
        if (b) {
          // Extra separation for Valve Cover vs Head vs Block vs Manifold
          let ox = spec.explodeOffset[0];
          let oy = spec.explodeOffset[1];
          let oz = spec.explodeOffset[2];
          if (nodeName === "Valve_Cover") {
            oy = 1.05;
            oz = 0;
          } else if (nodeName === "Cylinder_Head") {
            oy = 0.45;
            oz = 0;
          } else if (nodeName === "Engine_Block") {
            oy = 0;
            oz = 0;
          } else if (nodeName === "Exhaust_Manifold") {
            oy = 0.15;
            oz = -0.85;
          } else if (nodeName === "Camshaft_Intake") {
            oy = 0.65;
            oz = 0.35;
          } else if (nodeName === "Camshaft_Exhaust") {
            oy = 0.65;
            oz = -0.35;
          }
          obj.position.set(
            b.x + ox * explodeAmount,
            b.y + oy * explodeAmount,
            b.z + oz * explodeAmount
          );
        }
      }

      // Skip combustion chamber material override (handled above)
      if (nodeName.startsWith("Combustion_Chamber_")) return;

      const isPredHighlight = isHighlightedForPrediction(spec.id, prediction, confidenceTier);
      const isSelected = selectedId === spec.id;
      const isHovered = hoveredId === spec.id;
      const isOuterHousing =
        nodeName === "Engine_Block" || nodeName === "Cylinder_Head" || nodeName === "Valve_Cover";

      obj.traverse((child) => {
        if (!(child as THREE.Mesh).isMesh) return;
        const mesh = child as THREE.Mesh;
        const mat = mesh.material as THREE.MeshStandardMaterial;
        if (!mat) return;

        // Wireframe / X-Ray / Metallic mode
        mat.wireframe = renderStyle === "blueprint";

        if (renderStyle === "xray" && isOuterHousing && !isSelected) {
          mat.transparent = true;
          mat.opacity = 0.22;
          mat.depthWrite = false;
          mat.color.set("#64748B");
          mat.metalness = 0.35;
          mat.roughness = 0.2;
        } else if (renderStyle === "metallic" && isOuterHousing && !isSelected) {
          // Semi-cutaway metallic housing so glowing internals remain visible
          mat.transparent = true;
          mat.opacity = nodeName === "Engine_Block" ? 0.52 : 0.78;
          mat.depthWrite = true;
          mat.color.set("#94A3B8");
          mat.metalness = 0.88;
          mat.roughness = 0.22;
        } else if (renderStyle === "blueprint") {
          mat.transparent = true;
          mat.opacity = isOuterHousing ? 0.38 : 0.85;
          mat.depthWrite = true;
          mat.color.set("#38BDF8");
        } else {
          mat.transparent = false;
          mat.opacity = 1.0;
          mat.depthWrite = true;
        }

        // Emissive highlight for Selection / Hover / ML Prediction
        if (isSelected) {
          mat.emissive.set("#38BDF8");
          mat.emissiveIntensity = 0.65 + 0.25 * pulse;
        } else if (isHovered) {
          mat.emissive.set("#38BDF8");
          mat.emissiveIntensity = 0.4;
        } else if (isPredHighlight) {
          const faultColor = prediction === "misfire" ? "#EF4444" : "#F59E0B";
          mat.emissive.set(faultColor);
          mat.emissiveIntensity = 0.45 + 0.45 * pulse;
        } else if (renderStyle === "blueprint") {
          mat.emissive.set("#0284C7");
          mat.emissiveIntensity = 0.25;
        } else {
          mat.emissive.set("#000000");
          mat.emissiveIntensity = 0;
        }
      });
    });
  });

  const handlePointerClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    let curr: THREE.Object3D | null = e.object;
    while (curr) {
      const spec = findComponentByNodeName(curr.name);
      if (spec) {
        onSelectComponent(spec.id);
        return;
      }
      curr = curr.parent;
    }
  };

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    let curr: THREE.Object3D | null = e.object;
    while (curr) {
      const spec = findComponentByNodeName(curr.name);
      if (spec) {
        onHoverComponent(spec.id);
        document.body.style.cursor = "pointer";
        return;
      }
      curr = curr.parent;
    }
  };

  const handlePointerOut = () => {
    onHoverComponent(null);
    document.body.style.cursor = "auto";
  };

  // Pin callouts for ML-highlighted subsystems
  const highlightedComponents = ENGINE_COMPONENTS.filter((c) =>
    isHighlightedForPrediction(c.id, prediction, confidenceTier)
  );

  return (
    <group>
      <primitive
        object={clonedScene}
        onClick={handlePointerClick}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      />

      {/* Dynamic Internal Combustion Point Light */}
      <pointLight
        position={[0, 0.55, 0]}
        intensity={prediction === "engine_knock" ? 3.5 : prediction === "misfire" ? 1.8 : 2.2}
        distance={3.5}
        color={
          prediction === "engine_knock"
            ? "#F59E0B"
            : prediction === "misfire"
              ? "#EF4444"
              : "#38BDF8"
        }
      />

      {/* 3D Floating Callout Badges for Highlighted Inspection Areas */}
      {highlightedComponents.slice(0, 3).map((comp) => (
        <Html
          key={comp.id}
          position={[
            comp.cameraTarget[0],
            comp.cameraTarget[1] + comp.explodeOffset[1] * explodeAmount + 0.28,
            comp.cameraTarget[2] + comp.explodeOffset[2] * explodeAmount,
          ]}
          center
          distanceFactor={6}
          zIndexRange={[10, 0]}
        >
          <button
            type="button"
            onClick={() => onSelectComponent(comp.id)}
            className={`px-2.5 py-1 rounded-full text-[11px] font-medium whitespace-nowrap border shadow-lg backdrop-blur-md flex items-center gap-1.5 transition-transform hover:scale-105 ${
              prediction === "misfire"
                ? "bg-[#7F1D1D]/90 border-[#EF4444] text-[#FECACA]"
                : "bg-[#78350F]/90 border-[#F59E0B] text-[#FDE68A]"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full animate-ping ${
                prediction === "misfire" ? "bg-[#EF4444]" : "bg-[#F59E0B]"
              }`}
            />
            <span>Inspect: {comp.name.replace(" (Cyl 1–4)", "")}</span>
          </button>
        </Html>
      ))}
    </group>
  );
}

/* ── Live 1-3-4-2 Firing Order & Pressure Trace Mini-Oscilloscope ──────────── */
function FiringTraceHud({
  prediction,
  crankDeg,
  rpm,
}: {
  prediction: EngineCondition;
  crankDeg: number;
  rpm: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle baseline grid
    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.72);
    ctx.lineTo(w, h * 0.72);
    ctx.stroke();

    // Draw 4 cylinder firing pulses across 0..720 deg (Firing order: 1 - 3 - 4 - 2)
    const centers = [0.12, 0.37, 0.62, 0.87];
    ctx.lineWidth = 2;
    ctx.strokeStyle =
      prediction === "misfire"
        ? "#EF4444"
        : prediction === "engine_knock"
          ? "#F59E0B"
          : "#22C55E";

    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const u = x / w;
      let yVal = 0;
      centers.forEach((c, idx) => {
        const d = (u - c) / 0.065;
        if (prediction === "misfire" && idx === 1) {
          // Cylinder 3 missing pulse (flatline dip)
          yVal += -0.08 * Math.exp(-d * d);
        } else if (prediction === "engine_knock" && (idx === 1 || idx === 2)) {
          // Knock high-frequency pressure oscillation on peak
          const env = Math.exp(-d * d);
          yVal += env * (0.85 + 0.22 * Math.sin(u * 190));
        } else {
          yVal += 0.78 * Math.exp(-d * d);
        }
      });

      const py = h * 0.72 - yVal * (h * 0.56);
      if (x === 0) ctx.moveTo(x, py);
      else ctx.lineTo(x, py);
    }
    ctx.stroke();

    // Current crank angle playhead
    const playX = ((crankDeg % 720) / 720) * w;
    ctx.strokeStyle = "#38BDF8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 4);
    ctx.lineTo(playX, h - 4);
    ctx.stroke();
  }, [prediction, crankDeg, rpm]);

  const firingOrder = [
    { cyl: 1, deg: "0°" },
    { cyl: 3, deg: "180°", fault: prediction === "misfire" },
    { cyl: 4, deg: "360°", knock: prediction === "engine_knock" },
    { cyl: 2, deg: "540°" },
  ];

  return (
    <div className="rounded-[10px] bg-[#070A12]/90 border border-[#1E293B] p-3 backdrop-blur-md w-full sm:w-[272px] space-y-2">
      <div className="flex items-center justify-between text-[11px] font-mono-data">
        <span className="text-[#94A3B8]">FIRING ORDER 1-3-4-2</span>
        <span
          className={
            prediction === "misfire"
              ? "text-[#F87171]"
              : prediction === "engine_knock"
                ? "text-[#FBBF24]"
                : "text-[#4ADE80]"
          }
        >
          {prediction === "misfire"
            ? "CYL 3 DROP"
            : prediction === "engine_knock"
              ? "KNOCK RING"
              : "NOMINAL"}
        </span>
      </div>

      <div className="grid grid-cols-4 gap-1 text-center font-mono-data text-[10px]">
        {firingOrder.map((f) => (
          <div
            key={f.cyl}
            className={`rounded-[4px] py-0.5 border ${
              f.fault
                ? "border-[#EF4444]/50 bg-[#EF4444]/15 text-[#FCA5A5]"
                : f.knock
                  ? "border-[#F59E0B]/50 bg-[#F59E0B]/15 text-[#FDE68A]"
                  : "border-[#1E293B] bg-[#0F172A] text-[#CBD5E1]"
            }`}
          >
            C{f.cyl} · {f.deg}
          </div>
        ))}
      </div>

      <canvas ref={canvasRef} width={248} height={54} className="w-full h-[54px] block" />
    </div>
  );
}

/* ── Main Exported 3D Engine Failure Lab Component ─────────────────────────── */
export function EngineLab3D({
  condition = "healthy",
  confidence = 0.85,
  confidenceTier = "high",
  compact = false,
  allowSimulationOverride = true,
}: {
  condition?: EngineCondition;
  confidence?: number;
  confidenceTier?: ConfidenceTier;
  compact?: boolean;
  allowSimulationOverride?: boolean;
}) {
  const [webglSupported, setWebglSupported] = useState(true);
  const [activeCondition, setActiveCondition] = useState<EngineCondition>(condition);
  const [renderStyle, setRenderStyle] = useState<RenderStyle>("xray");
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>("hero");
  const [explodeAmount, setExplodeAmount] = useState(0);
  const [rpm, setRpm] = useState(1150);
  const [playing, setPlaying] = useState(true);
  const [crankDeg, setCrankDeg] = useState(180);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [showBrowserTree, setShowBrowserTree] = useState(!compact);

  const controlsRef = useRef<any>(null);

  useEffect(() => {
    setWebglSupported(checkWebGLAvailable());
  }, []);

  // Sync with external ML prediction whenever `condition` prop changes
  useEffect(() => {
    setActiveCondition(condition);
    if (condition === "misfire") {
      setSelectedId("spark_plugs");
    } else if (condition === "engine_knock") {
      setSelectedId("combustion_chamber");
    } else {
      setSelectedId(null);
    }
  }, [condition]);

  const selectedSpec = useMemo(
    () => ENGINE_COMPONENTS.find((c) => c.id === selectedId) || null,
    [selectedId]
  );

  const highlightedSpecs = useMemo(
    () =>
      ENGINE_COMPONENTS.filter((c) =>
        isHighlightedForPrediction(c.id, activeCondition, confidenceTier)
      ),
    [activeCondition, confidenceTier]
  );

  const handleResetView = () => {
    setCameraPreset("hero");
    setExplodeAmount(0);
    setSelectedId(null);
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  const handleZoom = (dir: "in" | "out") => {
    const ctrl = controlsRef.current;
    if (!ctrl || !ctrl.object) return;
    const cam = ctrl.object as THREE.PerspectiveCamera;
    const factor = dir === "in" ? 0.82 : 1.22;
    cam.position.multiplyScalar(factor);
    ctrl.update();
  };

  return (
    <div className="rounded-[12px] border border-[var(--border)] bg-[#070A12] text-[#F8FAFC] overflow-hidden shadow-2xl">
      {/* ── Top CAD Cockpit Toolbar ──────────────────────────────────────────── */}
      <div className="px-4 py-3 bg-[#0D1322] border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowBrowserTree((v) => !v)}
            className={`px-2.5 py-1.5 rounded-[6px] text-[12px] font-medium border flex items-center gap-1.5 transition-colors ${
              showBrowserTree
                ? "bg-[#0EA5E9]/15 border-[#38BDF8]/40 text-[#38BDF8]"
                : "bg-[#111827] border-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC]"
            }`}
          >
            <Layers size={14} strokeWidth={1.5} />
            <span>Assemblies ({ENGINE_COMPONENTS.length})</span>
          </button>

          {/* Render Style Selector (Cutaway X-Ray / Metallic PBR / Blueprint CAD) */}
          <div className="inline-flex items-center rounded-[8px] bg-[#070A12] border border-[#1E293B] p-0.5">
            {(
              [
                { id: "xray", label: "Cutaway X-Ray", icon: Eye },
                { id: "metallic", label: "Metallic PBR", icon: Sparkles },
                { id: "blueprint", label: "Blueprint CAD", icon: Cpu },
              ] as const
            ).map((mode) => {
              const Icon = mode.icon;
              const active = renderStyle === mode.id;
              return (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() => setRenderStyle(mode.id)}
                  className={`px-2.5 py-1 rounded-[6px] text-[12px] font-medium flex items-center gap-1.5 transition-colors ${
                    active
                      ? "bg-[#1E293B] text-[#38BDF8] border border-[#38BDF8]/30"
                      : "text-[#94A3B8] hover:text-[#F8FAFC]"
                  }`}
                >
                  <Icon size={13} strokeWidth={1.5} />
                  <span className="hidden sm:inline">{mode.label}</span>
                </button>
              );
            })}
          </div>

          {/* Exploded View Control */}
          <div className="flex items-center gap-2 bg-[#070A12] border border-[#1E293B] rounded-[8px] px-2.5 py-1">
            <button
              type="button"
              onClick={() => setExplodeAmount((v) => (v > 0.1 ? 0 : 0.75))}
              className={`text-[12px] font-medium flex items-center gap-1 ${
                explodeAmount > 0.05 ? "text-[#38BDF8]" : "text-[#94A3B8] hover:text-[#F8FAFC]"
              }`}
            >
              <Maximize2 size={13} strokeWidth={1.5} />
              <span>Explode</span>
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.02}
              value={explodeAmount}
              onChange={(e) => setExplodeAmount(Number(e.target.value))}
              className="studio-slider w-16 sm:w-20"
              aria-label="Exploded view distance"
            />
            <span className="font-mono-data text-[11px] text-[#94A3B8] w-8 text-right">
              {Math.round(explodeAmount * 100)}%
            </span>
          </div>
        </div>

        {/* ML Class Highlight Filter / Interactive Mode Switcher */}
        <div className="flex items-center gap-2 flex-wrap">
          {allowSimulationOverride && (
            <div className="inline-flex items-center rounded-[8px] bg-[#070A12] border border-[#1E293B] p-0.5 text-[12px]">
              {(
                [
                  { id: "healthy", label: "Healthy", dot: "bg-[#22C55E]" },
                  { id: "misfire", label: "Misfire", dot: "bg-[#EF4444]" },
                  { id: "engine_knock", label: "Engine Knocking", dot: "bg-[#F59E0B]" },
                ] as const
              ).map((cls) => {
                const active = activeCondition === cls.id;
                return (
                  <button
                    key={cls.id}
                    type="button"
                    onClick={() => {
                      setActiveCondition(cls.id);
                      if (cls.id === "misfire") setSelectedId("spark_plugs");
                      else if (cls.id === "engine_knock") setSelectedId("combustion_chamber");
                      else setSelectedId(null);
                    }}
                    className={`px-2.5 py-1 rounded-[6px] font-medium flex items-center gap-1.5 transition-colors ${
                      active
                        ? "bg-[#1E293B] text-[#F8FAFC] border border-[#334155]"
                        : "text-[#94A3B8] hover:text-[#F8FAFC]"
                    }`}
                  >
                    <span className={`h-2 w-2 rounded-full ${cls.dot}`} />
                    <span>{cls.label}</span>
                  </button>
                );
              })}
            </div>
          )}

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => handleZoom("in")}
              className="p-1.5 rounded-[6px] bg-[#111827] border border-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC]"
              aria-label="Zoom in 3D engine"
              title="Zoom in"
            >
              <ZoomIn size={14} strokeWidth={1.5} />
            </button>
            <button
              type="button"
              onClick={() => handleZoom("out")}
              className="p-1.5 rounded-[6px] bg-[#111827] border border-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC]"
              aria-label="Zoom out 3D engine"
              title="Zoom out"
            >
              <ZoomOut size={14} strokeWidth={1.5} />
            </button>
            <button
              type="button"
              onClick={handleResetView}
              className="px-2.5 py-1.5 rounded-[6px] bg-[#111827] border border-[#1E293B] text-[12px] text-[#94A3B8] hover:text-[#F8FAFC] flex items-center gap-1"
              title="Reset 3D camera and assembly"
            >
              <RotateCcw size={13} strokeWidth={1.5} />
              <span className="hidden md:inline">Reset</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Main 3D Viewport + CAD Model Browser + Inspection Drawer ────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 relative">
        {/* Left: CAD Assembly Browser Tree (Reference 4 inspired) */}
        {showBrowserTree && (
          <div className="lg:col-span-3 bg-[#0B101D] border-b lg:border-b-0 lg:border-r border-[#1E293B] p-3 flex flex-col justify-between max-h-[540px] overflow-y-auto">
            <div className="space-y-3">
              <div className="flex items-center justify-between px-1">
                <span className="text-[11px] font-mono-data uppercase tracking-wider text-[#94A3B8]">
                  Model Browser · GLB
                </span>
                <span className="text-[11px] font-mono-data text-[#38BDF8]">36 Meshes</span>
              </div>

              {/* Active ML Prediction Highlight Banner */}
              <div
                className={`rounded-[8px] border p-2.5 text-[12px] ${
                  activeCondition === "misfire"
                    ? "border-[#EF4444]/40 bg-[#EF4444]/10 text-[#FECACA]"
                    : activeCondition === "engine_knock"
                      ? "border-[#F59E0B]/40 bg-[#F59E0B]/10 text-[#FDE68A]"
                      : "border-[#22C55E]/35 bg-[#22C55E]/10 text-[#BBF7D0]"
                }`}
              >
                <div className="font-semibold flex items-center justify-between">
                  <span>ML Class: {predictionToLabel(activeCondition)}</span>
                  <span className="font-mono-data">{Math.round(confidence * 100)}%</span>
                </div>
                <p className="text-[11px] opacity-90 mt-0.5">
                  {activeCondition === "misfire"
                    ? "Highlighting spark plugs, ignition coils, and fuel injectors for inspection."
                    : activeCondition === "engine_knock"
                      ? "Highlighting combustion chambers, pistons, rods, crankshaft, and oil pan."
                      : "No fault highlights active. All 4 cylinders firing normally."}
                </p>
              </div>

              <div className="space-y-1">
                {ENGINE_COMPONENTS.map((comp) => {
                  const isHighlighted = isHighlightedForPrediction(
                    comp.id,
                    activeCondition,
                    confidenceTier
                  );
                  const isSelected = selectedId === comp.id;

                  return (
                    <button
                      key={comp.id}
                      type="button"
                      onClick={() => setSelectedId(isSelected ? null : comp.id)}
                      onMouseEnter={() => setHoveredId(comp.id)}
                      onMouseLeave={() => setHoveredId(null)}
                      className={`w-full px-2.5 py-2 rounded-[7px] text-left text-[12px] transition-all flex items-center justify-between gap-2 border ${
                        isSelected
                          ? "bg-[#0EA5E9]/20 border-[#38BDF8] text-[#F8FAFC]"
                          : isHighlighted
                            ? activeCondition === "misfire"
                              ? "bg-[#EF4444]/12 border-[#EF4444]/40 text-[#FECACA] hover:bg-[#EF4444]/20"
                              : "bg-[#F59E0B]/12 border-[#F59E0B]/40 text-[#FDE68A] hover:bg-[#F59E0B]/20"
                            : "bg-transparent border-transparent text-[#CBD5E1] hover:bg-[#1E293B]/60"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <Box
                          size={13}
                          strokeWidth={1.5}
                          className={
                            isSelected
                              ? "text-[#38BDF8] shrink-0"
                              : isHighlighted
                                ? activeCondition === "misfire"
                                  ? "text-[#F87171] shrink-0"
                                  : "text-[#FBBF24] shrink-0"
                                : "text-[#64748B] shrink-0"
                          }
                        />
                        <span className="truncate font-medium">{comp.name}</span>
                      </div>

                      {isHighlighted ? (
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-mono-data shrink-0 ${
                            activeCondition === "misfire"
                              ? "bg-[#EF4444]/25 text-[#FCA5A5]"
                              : "bg-[#F59E0B]/25 text-[#FDE68A]"
                          }`}
                        >
                          Inspect
                        </span>
                      ) : (
                        <span className="font-mono-data text-[10px] text-[#64748B]">
                          {comp.partCount}x
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="pt-3 border-t border-[#1E293B] text-[11px] text-[#64748B] flex items-center justify-between font-mono-data">
              <span>ASSET: I4_DOHC.GLB</span>
              <span>SCALE 1:1</span>
            </div>
          </div>
        )}

        {/* Center: Interactive Three.js WebGL Viewport */}
        <div
          className={`${
            showBrowserTree
              ? selectedSpec
                ? "lg:col-span-5"
                : "lg:col-span-9"
              : selectedSpec
                ? "lg:col-span-8"
                : "lg:col-span-12"
          } relative ${compact ? "h-[430px]" : "h-[540px]"} bg-[radial-gradient(ellipse_at_center,#131E36_0%,#070A12_78%)] select-none`}
        >
          {webglSupported ? (
            <Suspense
              fallback={
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-[#94A3B8]">
                  <Activity size={24} className="animate-spin text-[#38BDF8]" />
                  <span className="text-[13px] font-mono-data">
                    Loading 3D GLB Engine Assembly (36 meshes)...
                  </span>
                </div>
              }
            >
              <Canvas
                shadows
                dpr={[1, 2]}
                camera={{ position: [3.1, 1.65, 3.55], fov: 38, near: 0.1, far: 100 }}
                onPointerMissed={() => setSelectedId(null)}
              >
                <color attach="background" args={["#070A12"]} />
                <fog attach="fog" args={["#070A12", 8, 22]} />

                {/* Studio Automotive Three-Point + Rim Lighting */}
                <ambientLight intensity={0.65} />
                <directionalLight
                  position={[5, 8, 6]}
                  intensity={2.2}
                  castShadow
                  shadow-mapSize-width={1024}
                  shadow-mapSize-height={1024}
                />
                <directionalLight position={[-6, 4, -4]} intensity={1.1} color="#38BDF8" />
                <directionalLight position={[0, -5, 4]} intensity={0.65} color="#F59E0B" />

                {/* Engineering Blueprint Floor Grid */}
                <gridHelper
                  args={[14, 28, "#1E293B", "#0F172A"]}
                  position={[0, -2.05, 0]}
                />

                <EngineGlbScene
                  prediction={activeCondition}
                  confidenceTier={confidenceTier}
                  renderStyle={renderStyle}
                  explodeAmount={explodeAmount}
                  rpm={rpm}
                  playing={playing}
                  selectedId={selectedId}
                  hoveredId={hoveredId}
                  onSelectComponent={setSelectedId}
                  onHoverComponent={setHoveredId}
                  onCrankAngleUpdate={setCrankDeg}
                />

                <ContactShadows
                  position={[0, -2.04, 0]}
                  opacity={0.5}
                  scale={10}
                  blur={2.2}
                  far={4}
                />

                <OrbitControls
                  ref={controlsRef}
                  enableDamping
                  dampingFactor={0.07}
                  minDistance={1.6}
                  maxDistance={9.5}
                  maxPolarAngle={Math.PI * 0.85}
                />

                <CameraRig
                  preset={cameraPreset}
                  selectedSpec={selectedSpec}
                  controlsRef={controlsRef}
                />
              </Canvas>
            </Suspense>
          ) : (
            /* Graceful Fallback if WebGL is disabled in browser */
            <div className="h-full flex flex-col items-center justify-center p-6 text-center space-y-4">
              <AlertTriangle size={28} className="text-[#FBBF24]" />
              <div>
                <h4 className="text-[16px] font-semibold text-[#F8FAFC]">
                  Interactive Subsystem Selector (WebGL Fallback Mode)
                </h4>
                <p className="text-[13px] text-[#94A3B8] max-w-md mt-1">
                  Hardware WebGL acceleration is unavailable in this display context. Select any engine subsystem below to inspect ML-linked repair guidance.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2 max-w-xl">
                {ENGINE_COMPONENTS.map((c) => (
                  <Button
                    key={c.id}
                    size="sm"
                    variant={selectedId === c.id ? "primary" : "secondary"}
                    onClick={() => setSelectedId(c.id)}
                  >
                    {c.name}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Top-Left Viewport Orientation & Active Hover Badge */}
          <div className="absolute top-3 left-3 pointer-events-none flex flex-col gap-1.5">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-[6px] bg-[#070A12]/85 border border-[#1E293B] backdrop-blur-md text-[11px] font-mono-data text-[#94A3B8]">
              <span className="h-2 w-2 rounded-full bg-[#38BDF8]" />
              <span>DRAG: ROTATE · SCROLL: ZOOM · CLICK PART: INSPECT</span>
            </div>
            {hoveredId && (
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[6px] bg-[#0EA5E9]/20 border border-[#38BDF8]/50 text-[12px] font-medium text-[#F8FAFC]">
                <span>{ENGINE_COMPONENTS.find((c) => c.id === hoveredId)?.name}</span>
              </div>
            )}
          </div>

          {/* Bottom-Left Live RPM & Crank Telemetry HUD (Reference 1 inspired) */}
          <div className="absolute bottom-14 left-3 rounded-[10px] bg-[#070A12]/90 border border-[#1E293B] p-3 backdrop-blur-md space-y-2 w-[210px]">
            <div className="flex items-baseline justify-between">
              <div>
                <span className="font-mono-data text-[24px] font-semibold text-[#F8FAFC] leading-none">
                  {playing ? rpm : 0}
                </span>
                <span className="font-mono-data text-[11px] text-[#94A3B8] ml-1">rpm</span>
              </div>
              <span className="font-mono-data text-[11px] text-[#38BDF8]">
                Crank {crankDeg}°
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                className="px-2 py-1 rounded-[5px] bg-[#1E293B] hover:bg-[#334155] text-[11px] font-mono-data text-[#F8FAFC] flex items-center gap-1"
              >
                {playing ? <Pause size={11} /> : <Play size={11} />}
                <span>{playing ? "PAUSE" : "PLAY"}</span>
              </button>
              <input
                type="range"
                min={600}
                max={2800}
                step={50}
                value={rpm}
                onChange={(e) => setRpm(Number(e.target.value))}
                className="studio-slider flex-1"
                aria-label="Engine RPM speed"
              />
            </div>
          </div>

          {/* Bottom-Right Live 1-3-4-2 Firing Trace HUD */}
          {!selectedSpec && (
            <div className="hidden sm:block absolute bottom-14 right-3">
              <FiringTraceHud prediction={activeCondition} crankDeg={crankDeg} rpm={rpm} />
            </div>
          )}

          {/* Bottom Camera Presets Bar (1 Hero · 2 Cranktrain · 3 Cylinders · 4 Valvetrain · 5 Combustion · 6 Ignition) */}
          <div className="absolute bottom-2.5 inset-x-3 flex items-center justify-center">
            <div className="inline-flex items-center gap-1 px-2 py-1 rounded-[8px] bg-[#070A12]/90 border border-[#1E293B] backdrop-blur-md overflow-x-auto max-w-full">
              <Camera size={13} className="text-[#94A3B8] mx-1 shrink-0" />
              {CAMERA_PRESETS.map((preset) => {
                const active = cameraPreset === preset.id && !selectedSpec;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => {
                      setSelectedId(null);
                      setCameraPreset(preset.id);
                    }}
                    className={`px-2 py-1 rounded-[5px] text-[11px] font-mono-data whitespace-nowrap transition-colors ${
                      active
                        ? "bg-[#0284C7] text-white"
                        : "text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
                    }`}
                  >
                    {preset.num} {preset.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Component Inspection & ML Correlation Drawer (Opens when a 3D component is clicked) */}
        {selectedSpec && (
          <div className="lg:col-span-4 bg-[#0B101D] border-t lg:border-t-0 lg:border-l border-[#1E293B] p-5 flex flex-col justify-between max-h-[540px] overflow-y-auto space-y-4">
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-2 border-b border-[#1E293B] pb-3">
                <div>
                  <span className="text-[11px] font-mono-data uppercase tracking-wider text-[#38BDF8]">
                    {selectedSpec.categoryLabel}
                  </span>
                  <h4 className="text-[17px] font-semibold text-[#F8FAFC] mt-0.5">
                    {selectedSpec.name}
                  </h4>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedId(null)}
                  className="text-[12px] text-[#94A3B8] hover:text-[#F8FAFC] px-2 py-1 rounded bg-[#111827] border border-[#1E293B]"
                >
                  Close
                </button>
              </div>

              {/* Status Badge relative to current ML prediction */}
              {isHighlightedForPrediction(selectedSpec.id, activeCondition, confidenceTier) ? (
                <div
                  className={`rounded-[8px] border p-3 space-y-1.5 text-[12px] ${
                    activeCondition === "misfire"
                      ? "border-[#EF4444]/45 bg-[#EF4444]/12 text-[#FECACA]"
                      : "border-[#F59E0B]/45 bg-[#F59E0B]/12 text-[#FDE68A]"
                  }`}
                >
                  <div className="font-semibold flex items-center gap-1.5">
                    <ShieldAlert size={15} strokeWidth={1.75} />
                    <span>Possible Inspection Area ({predictionToLabel(activeCondition)})</span>
                  </div>
                  <p className="leading-relaxed">
                    {activeCondition === "misfire"
                      ? selectedSpec.misfireRelevance
                      : selectedSpec.knockRelevance}
                  </p>
                  <p className="text-[11px] opacity-80 pt-1 border-t border-current/20">
                    Audio classification highlights candidate subsystems for inspection — it never confirms a specific component is faulty without physical testing.
                  </p>
                </div>
              ) : (
                <div className="rounded-[8px] border border-[#1E293B] bg-[#111827] p-3 text-[12px] text-[#CBD5E1] flex items-center gap-2">
                  <CheckCircle2 size={15} className="text-[#4ADE80] shrink-0" />
                  <span>
                    Not flagged as a primary inspection area for the current{" "}
                    <strong className="text-[#F8FAFC]">{predictionToLabel(activeCondition)}</strong>{" "}
                    prediction.
                  </span>
                </div>
              )}

              {/* Engineering Metadata */}
              <div className="grid grid-cols-2 gap-2 text-[12px] font-mono-data">
                <div className="rounded-[6px] bg-[#111827] border border-[#1E293B] p-2.5">
                  <div className="text-[10px] text-[#64748B]">QUANTITY</div>
                  <div className="text-[#F8FAFC] font-semibold mt-0.5">
                    {selectedSpec.partCount} {selectedSpec.partCount === 1 ? "Assembly" : "Units"}
                  </div>
                </div>
                <div className="rounded-[6px] bg-[#111827] border border-[#1E293B] p-2.5">
                  <div className="text-[10px] text-[#64748B]">INSPECTION STATUS</div>
                  <div className="text-[#38BDF8] font-semibold mt-0.5">
                    {isHighlightedForPrediction(selectedSpec.id, activeCondition, confidenceTier)
                      ? "Priority Check"
                      : "Routine"}
                  </div>
                </div>
              </div>

              <div className="space-y-1.5 text-[13px]">
                <div className="text-[12px] font-semibold text-[#94A3B8]">Mechanical Function</div>
                <p className="text-[#CBD5E1] leading-relaxed">{selectedSpec.functionSummary}</p>
              </div>

              <div className="space-y-1.5 text-[13px]">
                <div className="text-[12px] font-semibold text-[#94A3B8]">Acoustic Signature</div>
                <p className="text-[#CBD5E1] leading-relaxed">{selectedSpec.acousticSignature}</p>
              </div>

              <div className="space-y-2">
                <div className="text-[12px] font-semibold text-[#94A3B8] flex items-center gap-1.5">
                  <Wrench size={13} className="text-[#38BDF8]" />
                  <span>Recommended Component Checks</span>
                </div>
                <ul className="space-y-1.5">
                  {selectedSpec.inspectionSteps.map((step, i) => (
                    <li
                      key={i}
                      className="rounded-[6px] bg-[#111827] border border-[#1E293B] px-3 py-2 text-[12px] text-[#F8FAFC] flex items-start gap-2"
                    >
                      <span className="font-mono-data text-[#38BDF8]">{i + 1}.</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="pt-3 border-t border-[#1E293B] text-[11px] text-[#94A3B8] flex items-center gap-1.5">
              <Info size={13} className="text-[#38BDF8] shrink-0" />
              <span>Material: {selectedSpec.materialSpec}</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Bottom Engineering Status Bar & Quick Subsystem Pills ────────────── */}
      <div className="px-4 py-2.5 bg-[#0D1322] border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-3 text-[12px]">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[#94A3B8] font-mono-data text-[11px]">
            HIGHLIGHTED INSPECTION ZONES:
          </span>
          {highlightedSpecs.length === 0 ? (
            <Badge tone="healthy" dot>
              Healthy — No Fault Highlights
            </Badge>
          ) : (
            highlightedSpecs.map((spec) => (
              <button
                key={spec.id}
                type="button"
                onClick={() => setSelectedId(spec.id)}
                className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium border transition-colors flex items-center gap-1.5 ${
                  selectedId === spec.id
                    ? "bg-[#38BDF8] text-[#070A12] border-[#38BDF8]"
                    : activeCondition === "misfire"
                      ? "bg-[#EF4444]/15 border-[#EF4444]/40 text-[#FCA5A5] hover:bg-[#EF4444]/25"
                      : "bg-[#F59E0B]/15 border-[#F59E0B]/40 text-[#FDE68A] hover:bg-[#F59E0B]/25"
                }`}
              >
                <Flame size={11} />
                <span>{spec.name.replace(" (Cyl 1–4)", "").replace(" (COP 1–4)", "")}</span>
              </button>
            ))
          )}
        </div>

        <div className="text-[11px] text-[#94A3B8]">
          Screening indicator only — physical inspection required to confirm any component fault.
        </div>
      </div>
    </div>
  );
}

useGLTF.preload("/models/autoguard_i4_engine.glb");
