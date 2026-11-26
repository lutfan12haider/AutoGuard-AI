import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import gsap from "gsap";
import * as THREE from "three";

const KNOCK_RED = new THREE.Color("#EF4444");
const MISFIRE_ORANGE = new THREE.Color("#F97316");
const ENGINE_MOUNT_Y = 0.585;

function buildStudioEnvironment(gl: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(gl);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color("#1E2429");

  const addPanel = (color: number, intensity: number, pos: [number, number, number], rot: [number, number, number], size: [number, number]) => {
    const geo = new THREE.PlaneGeometry(size[0], size[1]);
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...pos);
    mesh.rotation.set(...rot);
    envScene.add(mesh);
  };

  // Pure neutral studio softboxes for true silver Toyota paint & clean aluminum highlights (no green tint)
  addPanel(0xffffff, 2.2, [0, 6, 1], [Math.PI / 2, 0, 0], [8, 11]);
  addPanel(0xe2e8f0, 1.3, [-6, 2.5, 2], [0, Math.PI / 2, 0], [10, 3.2]);
  addPanel(0xf1f5f9, 1.3, [6, 2.5, 2], [0, -Math.PI / 2, 0], [10, 3.2]);
  addPanel(0xcbd5e1, 0.9, [0, 2.2, 6.5], [0, Math.PI, 0], [7, 2.4]);
  addPanel(0x1f2933, 0.8, [0, -3.5, 0], [-Math.PI / 2, 0, 0], [14, 14]);

  const rt = pmrem.fromScene(envScene, 0.035);
  envScene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    if (m.material && !Array.isArray(m.material)) m.material.dispose();
  });
  pmrem.dispose();
  return rt.texture;
}

function createSoftShadowTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext("2d")!;
  const grad = ctx.createRadialGradient(128, 128, 28, 128, 128, 124);
  grad.addColorStop(0, "rgba(0,0,0,0.68)");
  grad.addColorStop(0.58, "rgba(0,0,0,0.34)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}
import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  Play,
  RotateCcw,
  ShieldAlert,
  Wrench,
} from "lucide-react";
import type { ConfidenceTier, EngineCondition } from "../types/diagnostic";
import { predictionToLabel } from "../services/api";
import { Badge, Button } from "./ui";

const CRANK_PHASES = [0.0, Math.PI, 3 * Math.PI, 2 * Math.PI];

type SequencePhaseId = "idle" | "establishing" | "approaching" | "opening_bonnet" | "revealing_engine" | "complete";

const PHASE_LABELS: Record<SequencePhaseId, string> = {
  idle: "Ready in Diagnostic Bay",
  establishing: "01 · Positioning Vehicle in Bay",
  approaching: "02 · Moving to Front Powertrain",
  opening_bonnet: "03 · Opening Vehicle Bonnet",
  revealing_engine: "04 · Revealing Internal Engine Assembly",
  complete: "05 · Diagnostic Telemetry Active",
};

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

/* ── React ErrorBoundary for Safe 3D Fallback ──────────────────────────────── */
class Scene3DErrorBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: ReactNode; fallback: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

/* ── Dual-GLB Sports Sedan + Animated Engine Core Scene ────────────────────── */
function DiagnosticBayScene({
  condition,
  confidence,
  confidenceTier,
  playToken,
  autoPlay,
  bonnetOpenTarget,
  cutawayMode,
  cameraViewPreset,
  controlsRef,
  onPhaseChange,
  onWarningVisible,
  onReady,
}: {
  condition: EngineCondition;
  confidence: number;
  confidenceTier: ConfidenceTier;
  playToken: number;
  autoPlay: boolean;
  bonnetOpenTarget: boolean;
  cutawayMode: boolean;
  cameraViewPreset: "sedan" | "front_bay" | "closeup" | null;
  controlsRef: React.RefObject<any>;
  onPhaseChange: (phase: SequencePhaseId) => void;
  onWarningVisible: (visible: boolean) => void;
  onReady: () => void;
}) {
  const { camera, gl, scene } = useThree();
  const sedanGltf = useGLTF("/models/autoguard_corolla.glb");
  const engineGltf = useGLTF("/models/autoguard_i4_engine.glb");

  const shadowTex = useMemo(() => createSoftShadowTexture(), []);

  useEffect(() => {
    const envTex = buildStudioEnvironment(gl);
    scene.environment = envTex;
    onReady();
    return () => {
      if (scene.environment === envTex) scene.environment = null;
      envTex.dispose();
    };
  }, [gl, scene, onReady]);

  useEffect(() => {
    return () => {
      shadowTex.dispose();
    };
  }, [shadowTex]);

  const sedanClone = useMemo(() => sedanGltf.scene.clone(true), [sedanGltf.scene]);
  const engineClone = useMemo(() => engineGltf.scene.clone(true), [engineGltf.scene]);

  const engineMountRef = useRef<THREE.Group>(null);
  const bayLightRef = useRef<THREE.PointLight>(null);
  const faultLightRef = useRef<THREE.PointLight>(null);
  const bonnetPivotRef = useRef<THREE.Object3D | null>(null);
  const timelineRef = useRef<gsap.core.Timeline | null>(null);

  const engineMaterialsRef = useRef<Array<{
    nodeName: string;
    mat: THREE.MeshStandardMaterial;
    origColor: THREE.Color;
    origOpacity: number;
    origTransparent: boolean;
  }>>([]);

  const sedanMaterialsRef = useRef<THREE.Material[]>([]);
  const crankRef = useRef(0);
  const animParamsRef = useRef({
    revealCutaway: 0,
    baySpot: 0.4,
    faultIntensity: 0,
  });

  // Collect & clone Sedan materials cleanly + find Bonnet_Hood_Pivot hinge
  useEffect(() => {
    const mats: THREE.Material[] = [];
    sedanClone.traverse((obj) => {
      if (obj.name === "Bonnet_Hood_Pivot") {
        bonnetPivotRef.current = obj;
      }
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        if (Array.isArray(mesh.material)) {
          mesh.material = mesh.material.map((m) => {
            const c = m.clone();
            mats.push(c);
            return c;
          });
        } else if (mesh.material) {
          const c = mesh.material.clone();
          mesh.material = c;
          mats.push(c);
        }
      }
    });
    sedanMaterialsRef.current = mats;

    return () => {
      mats.forEach((m) => m.dispose());
    };
  }, [sedanClone]);

  // Collect & clone Engine materials + store kinematic rest coordinates
  const engineRestRef = useRef<
    Map<string, { y: number; x: number; z: number; obj: THREE.Object3D }>
  >(new Map());

  useEffect(() => {
    const entries: typeof engineMaterialsRef.current = [];
    const restMap = new Map<string, { y: number; x: number; z: number; obj: THREE.Object3D }>();

    engineClone.traverse((obj) => {
      restMap.set(obj.name, { x: obj.position.x, y: obj.position.y, z: obj.position.z, obj });
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const srcMat = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial;
      if (!srcMat) return;
      const cloned = srcMat.clone();
      mesh.material = cloned;
      entries.push({
        nodeName: obj.name,
        mat: cloned,
        origColor: cloned.color.clone(),
        origOpacity: cloned.opacity,
        origTransparent: cloned.transparent,
      });
    });

    engineMaterialsRef.current = entries;
    engineRestRef.current = restMap;

    return () => {
      entries.forEach((e) => e.mat.dispose());
    };
  }, [engineClone]);

  // Helper to check if a given engine GLB node belongs to active diagnostic fault
  const isNodeFaultTarget = (nodeName: string): boolean => {
    if (confidenceTier === "low" || confidenceTier === "unsupported" || condition === "uncertain") {
      return false;
    }
    if (condition === "engine_knock") {
      return (
        nodeName.startsWith("Piston_") ||
        nodeName.startsWith("Connecting_Rod_") ||
        nodeName.startsWith("Combustion_Chamber_") ||
        nodeName === "Crankshaft" ||
        nodeName === "Oil_Pan"
      );
    }
    if (condition === "misfire") {
      return (
        nodeName.startsWith("Spark_Plug_") ||
        nodeName.startsWith("Ignition_Coil_") ||
        nodeName.startsWith("Fuel_Injector_") ||
        nodeName === "Fuel_Rail"
      );
    }
    if (condition === "healthy") {
      return nodeName.startsWith("Camshaft_") || nodeName === "Valve_Cover";
    }
    return false;
  };

  // Trigger full 5-stage GSAP Cinematic Diagnostic Sequence
  useEffect(() => {
    const bonnet = bonnetPivotRef.current;
    const ctrl = controlsRef.current;
    timelineRef.current?.kill();

    if (!autoPlay) {
      if (bonnet) bonnet.rotation.x = bonnetOpenTarget ? -0.88 : 0;
      animParamsRef.current.revealCutaway = cutawayMode ? 1 : 0;
      animParamsRef.current.baySpot = bonnetOpenTarget ? 1.0 : 0.4;
      animParamsRef.current.faultIntensity = 1.0;
      onPhaseChange("complete");
      onWarningVisible(true);
      return;
    }

    onWarningVisible(false);
    onPhaseChange("establishing");

    // Reset initial bay frame
    if (bonnet) bonnet.rotation.x = 0;
    animParamsRef.current.revealCutaway = 0;
    animParamsRef.current.baySpot = 0.4;
    animParamsRef.current.faultIntensity = 0;

    camera.position.set(3.85, 1.68, 4.10);
    const camTarget = ctrl ? ctrl.target : new THREE.Vector3(0, 0.64, 0.20);
    camTarget.set(0.0, 0.64, 0.20);
    ctrl?.update();

    const tl = gsap.timeline({
      defaults: { ease: "power2.inOut" },
    });

    // Stage 1 -> 2: Glide camera smoothly toward front Corolla grille & hood
    tl.to(
      camera.position,
      {
        x: 1.55,
        y: 1.48,
        z: 3.25,
        duration: 1.45,
        onStart: () => onPhaseChange("approaching"),
        onUpdate: () => ctrl?.update(),
      },
      0.45
    );
    tl.to(
      camTarget,
      {
        x: 0.0,
        y: 0.72,
        z: 1.62,
        duration: 1.45,
        onUpdate: () => ctrl?.update(),
      },
      0.45
    );

    // Stage 3: Smoothly open Corolla bonnet along windshield cowl hinge
    if (bonnet) {
      tl.to(
        bonnet.rotation,
        {
          x: -0.88,
          duration: 1.25,
          ease: "power2.out",
          onStart: () => onPhaseChange("opening_bonnet"),
        },
        1.65
      );
    }
    tl.to(
      animParamsRef.current,
      {
        baySpot: 1.0,
        duration: 1.0,
        ease: "power1.out",
      },
      1.7
    );

    // Stage 4: Settle closely over open Corolla engine bay & reveal detailed cutaway engine
    tl.to(
      camera.position,
      {
        x: 0.95,
        y: 1.62,
        z: 2.48,
        duration: 1.25,
        onStart: () => onPhaseChange("revealing_engine"),
        onUpdate: () => ctrl?.update(),
      },
      2.65
    );
    tl.to(
      camTarget,
      {
        x: 0.0,
        y: 0.62,
        z: 1.69,
        duration: 1.25,
        onUpdate: () => ctrl?.update(),
      },
      2.65
    );
    tl.to(
      animParamsRef.current,
      {
        revealCutaway: 1.0,
        duration: 0.8,
        ease: "power1.inOut",
      },
      2.9
    );

    // Stage 5: Illuminate fault region & present animated warning card
    tl.to(
      animParamsRef.current,
      {
        faultIntensity: 1.0,
        duration: 0.65,
        ease: "power2.out",
        onStart: () => {
          onPhaseChange("complete");
          onWarningVisible(true);
        },
      },
      3.65
    );

    timelineRef.current = tl;
    return () => {
      tl.kill();
    };
  }, [playToken, autoPlay]);

  // Manual bonnet toggle outside initial sequence
  useEffect(() => {
    const bonnet = bonnetPivotRef.current;
    if (!bonnet || (timelineRef.current && timelineRef.current.isActive())) return;
    gsap.to(bonnet.rotation, {
      x: bonnetOpenTarget ? -0.88 : 0,
      duration: 0.85,
      ease: "power2.out",
    });
    gsap.to(animParamsRef.current, {
      baySpot: bonnetOpenTarget ? 1.0 : 0.4,
      duration: 0.6,
    });
  }, [bonnetOpenTarget]);

  // Manual cutaway toggle outside active timeline
  useEffect(() => {
    if (timelineRef.current && timelineRef.current.isActive()) return;
    gsap.to(animParamsRef.current, {
      revealCutaway: cutawayMode ? 1.0 : 0.0,
      duration: 0.55,
      ease: "power1.inOut",
    });
  }, [cutawayMode]);

  // Camera view quick presets
  useEffect(() => {
    if (!cameraViewPreset) return;
    const ctrl = controlsRef.current;
    const target = ctrl ? ctrl.target : new THREE.Vector3();
    timelineRef.current?.kill();

    const views = {
      sedan: { pos: [3.85, 1.68, 4.10], tgt: [0.0, 0.64, 0.20] },
      front_bay: { pos: [1.55, 1.48, 3.25], tgt: [0.0, 0.72, 1.62] },
      closeup: { pos: [0.95, 1.62, 2.48], tgt: [0.0, 0.62, 1.69] },
    }[cameraViewPreset];

    gsap.to(camera.position, {
      x: views.pos[0],
      y: views.pos[1],
      z: views.pos[2],
      duration: 0.9,
      ease: "power2.inOut",
      onUpdate: () => ctrl?.update(),
    });
    gsap.to(target, {
      x: views.tgt[0],
      y: views.tgt[1],
      z: views.tgt[2],
      duration: 0.9,
      ease: "power2.inOut",
      onUpdate: () => ctrl?.update(),
    });
  }, [cameraViewPreset]);

  // Per-frame engine kinematics, knock vibration, misfire stumble & emissive pulses
  useFrame((state, delta) => {
    const dt = Math.min(delta, 0.05);
    const t = state.clock.elapsedTime;
    const faultMul = animParamsRef.current.faultIntensity;
    const isLowConf =
      confidenceTier === "low" || confidenceTier === "unsupported" || condition === "uncertain";

    // Crank angular velocity
    let omega = 18.0;
    if (!isLowConf && condition === "misfire") {
      // Subtle rhythmic stutter at cylinder 3 combustion window
      const stumble = Math.sin(crankRef.current * 0.5 + 1.2) > 0.65 ? 0.42 : 1.0;
      omega = 15.5 * stumble;
    } else if (!isLowConf && condition === "engine_knock") {
      omega = 20.5;
    }
    crankRef.current += dt * omega;
    const crankPhase = crankRef.current;

    const rest = engineRestRef.current;
    const crankNode = rest.get("Crankshaft");
    if (crankNode) crankNode.obj.rotation.x = crankPhase;
    const camIn = rest.get("Camshaft_Intake");
    if (camIn) camIn.obj.rotation.x = crankPhase * 0.5;
    const camEx = rest.get("Camshaft_Exhaust");
    if (camEx) camEx.obj.rotation.x = crankPhase * 0.5;

    for (let i = 0; i < 4; i++) {
      const cNum = i + 1;
      const ph = crankPhase + CRANK_PHASES[i];
      const dy = (Math.cos(ph) - Math.cos(CRANK_PHASES[i])) * 0.24;
      const pNode = rest.get(`Piston_${cNum}`);
      const rNode = rest.get(`Connecting_Rod_${cNum}`);
      if (pNode) pNode.obj.position.y = pNode.y + dy;
      if (rNode) {
        rNode.obj.position.y = rNode.y + dy;
        rNode.obj.rotation.x = Math.sin(ph) * 0.14;
      }
    }

    // Subtle knock vibration on engine mount during engine_knock
    if (engineMountRef.current) {
      if (!isLowConf && condition === "engine_knock" && faultMul > 0.1) {
        const amp = 0.0055 * faultMul * (0.65 + 0.35 * confidence);
        engineMountRef.current.position.x = Math.sin(t * 56) * amp;
        engineMountRef.current.position.y = ENGINE_MOUNT_Y + Math.cos(t * 72) * amp * 0.8;
      } else if (!isLowConf && condition === "misfire" && faultMul > 0.1) {
        const lope = Math.sin(t * 14) * 0.003 * faultMul;
        engineMountRef.current.position.x = lope;
        engineMountRef.current.position.y = ENGINE_MOUNT_Y;
      } else {
        engineMountRef.current.position.x = 0;
        engineMountRef.current.position.y = ENGINE_MOUNT_Y;
      }
    }

    // Smoothly suppress internal bay fault lights when bonnet is closed (zero bleed!)
    const bonnetAngle = bonnetPivotRef.current?.rotation.x ?? -0.88;
    const bonnetOpenFactor = Math.max(0, Math.min(1, (-bonnetAngle - 0.12) / 0.45));
    const effectiveFaultMul = faultMul * bonnetOpenFactor;

    // Engine bay illumination & fault light
    if (bayLightRef.current) {
      bayLightRef.current.intensity = 0.95 * animParamsRef.current.baySpot * (0.25 + 0.75 * bonnetOpenFactor);
    }
    if (faultLightRef.current) {
      const pulse = 0.5 + 0.5 * Math.sin(t * (condition === "engine_knock" ? 8.5 : 5.2));
      if (isLowConf || effectiveFaultMul < 0.04) {
        faultLightRef.current.intensity = 0;
      } else if (condition === "engine_knock") {
        faultLightRef.current.color.set("#EF4444");
        faultLightRef.current.intensity = (1.8 + 1.8 * pulse) * effectiveFaultMul;
      } else if (condition === "misfire") {
        faultLightRef.current.color.set("#F97316");
        faultLightRef.current.intensity = (1.6 + 1.6 * pulse) * effectiveFaultMul;
      } else {
        faultLightRef.current.color.set("#22C55E");
        faultLightRef.current.intensity = 0.65 * effectiveFaultMul;
      }
    }

    // Cutaway translucency + vivid color-coded fault highlights
    const cut = animParamsRef.current.revealCutaway;
    const pulse = 0.5 + 0.5 * Math.sin(t * (condition === "engine_knock" ? 8.0 : 5.0));

    engineMaterialsRef.current.forEach(({ nodeName, mat, origColor, origOpacity, origTransparent }) => {
      const isOuterShell =
        nodeName === "Valve_Cover" ||
        nodeName === "Cylinder_Head" ||
        nodeName === "Engine_Block";

      const shellFade = condition === "healthy" ? 0.64 : 0.79;
      if (isOuterShell && cut > 0.05) {
        mat.transparent = true;
        mat.opacity = 1.0 - shellFade * cut;
        mat.depthWrite = cut < 0.25;
      } else {
        mat.transparent = origTransparent;
        mat.opacity = origOpacity;
        mat.depthWrite = true;
      }

      if (isNodeFaultTarget(nodeName) && effectiveFaultMul > 0.04) {
        if (condition === "engine_knock") {
          mat.color.copy(origColor).lerp(KNOCK_RED, 0.76 * effectiveFaultMul);
          mat.emissive.set("#DC2626");
          mat.emissiveIntensity = (0.95 + 0.85 * pulse) * effectiveFaultMul;
        } else if (condition === "misfire") {
          const cyl3Boost = nodeName.endsWith("_3") ? 1.25 : 0.90;
          mat.color.copy(origColor).lerp(MISFIRE_ORANGE, 0.80 * effectiveFaultMul);
          mat.emissive.set(nodeName.endsWith("_3") ? "#EA580C" : "#F97316");
          mat.emissiveIntensity = (0.95 + 0.85 * pulse) * cyl3Boost * effectiveFaultMul;
        } else {
          mat.color.copy(origColor);
          mat.emissive.set("#16A34A");
          mat.emissiveIntensity = 0.22 * effectiveFaultMul;
        }
      } else {
        if (isOuterShell && !isLowConf && (condition === "engine_knock" || condition === "misfire") && cut > 0.2) {
          mat.color.copy(origColor).multiplyScalar(0.58);
        } else {
          mat.color.copy(origColor);
        }
        mat.emissive.set("#000000");
        mat.emissiveIntensity = 0;
      }
    });
  });

  return (
    <group>
      {/* Architectural Emerald Laboratory Floor & Turntable Plinth */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]} receiveShadow>
        <circleGeometry args={[12, 48]} />
        <meshStandardMaterial color="#0A1E15" roughness={0.78} metalness={0.22} />
      </mesh>
      <mesh position={[0, 0.012, 0]} receiveShadow>
        <cylinderGeometry args={[3.15, 3.25, 0.035, 64]} />
        <meshStandardMaterial color="#10271C" roughness={0.48} metalness={0.40} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.032, 0]}>
        <ringGeometry args={[3.06, 3.12, 64]} />
        <meshBasicMaterial color="#22C55E" opacity={0.70} transparent />
      </mesh>

      {/* Overhead Architectural Studio Softbox Strips (Clean metallic vehicle reflections) */}
      <mesh position={[0, 3.85, 0.6]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1.2, 4.4]} />
        <meshBasicMaterial color="#F8FAFC" />
      </mesh>
      <mesh position={[-1.9, 3.6, 0.4]} rotation={[Math.PI / 2, 0.25, 0]}>
        <planeGeometry args={[0.45, 3.8]} />
        <meshBasicMaterial color="#E2E8F0" />
      </mesh>
      <mesh position={[1.9, 3.6, 0.4]} rotation={[Math.PI / 2, -0.25, 0]}>
        <planeGeometry args={[0.45, 3.8]} />
        <meshBasicMaterial color="#E2E8F0" />
      </mesh>

      {/* Soft Architectural Undercar Ground Shadow (Zero sub-root portal overhead) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.032, 0]}>
        <planeGeometry args={[3.4, 5.8]} />
        <meshBasicMaterial map={shadowTex} transparent depthWrite={false} />
      </mesh>

      {/* 2023 Toyota Corolla Hybrid Metallic Silver */}
      <primitive object={sedanClone} />

      {/* 3D 4-Cylinder DOHC Engine Mounted Cleanly Inside Corolla Front Engine Bay */}
      <group
        ref={engineMountRef}
        position={[0, ENGINE_MOUNT_Y, 1.67]}
        rotation={[0, -Math.PI / 2, 0]}
        scale={0.148}
      >
        <primitive object={engineClone} />
      </group>

      {/* Engine Bay Studio Inspection Light & Fault Accent Light (strictly inside bay below closed hood) */}
      <pointLight ref={bayLightRef} position={[0, 1.38, 1.68]} intensity={0.4} distance={4.0} color="#F8FAFC" />
      <pointLight ref={faultLightRef} position={[0, 0.72, 1.67]} intensity={0} distance={2.2} color="#16A34A" />

      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.07}
        minDistance={1.55}
        maxDistance={7.5}
        maxPolarAngle={Math.PI * 0.48}
        onStart={() => {
          timelineRef.current?.kill();
          gsap.killTweensOf(camera.position);
          if (controlsRef.current?.target) {
            gsap.killTweensOf(controlsRef.current.target);
          }
          onWarningVisible(true);
          onPhaseChange("complete");
        }}
      />
    </group>
  );
}

/* ── Exported Responsive 3D Vehicle Diagnostic Card ────────────────────────── */
export function VehicleDiagnosticScene3D({
  condition = "healthy",
  confidence = 0.85,
  confidenceTier = "high",
  sequenceKey = "initial",
  autoPlay = true,
  compact = false,
  onInspectSubsystem,
}: {
  condition?: EngineCondition;
  confidence?: number;
  confidenceTier?: ConfidenceTier;
  sequenceKey?: string | number;
  autoPlay?: boolean;
  compact?: boolean;
  onInspectSubsystem?: () => void;
}) {
  const [webglOk, setWebglOk] = useState(true);
  const [phase, setPhase] = useState<SequencePhaseId>(autoPlay ? "establishing" : "complete");
  const [warningVisible, setWarningVisible] = useState(!autoPlay);
  const [playToken, setPlayToken] = useState(0);
  const [manualReplay, setManualReplay] = useState(autoPlay);
  const [bonnetOpen, setBonnetOpen] = useState(!autoPlay);
  const [cutawayMode, setCutawayMode] = useState(!autoPlay);
  const [cameraView, setCameraView] = useState<"sedan" | "front_bay" | "closeup" | null>(null);
  const [sceneReady, setSceneReady] = useState(false);

  const controlsRef = useRef<any>(null);

  useEffect(() => {
    setWebglOk(checkWebGLAvailable());
  }, []);

  // Re-trigger automated GSAP sequence whenever a new prediction arrives
  useEffect(() => {
    if (!autoPlay) return;
    setManualReplay(true);
    setBonnetOpen(true);
    setCutawayMode(true);
    setCameraView(null);
    setPlayToken((k) => k + 1);
  }, [sequenceKey, condition, autoPlay]);

  const isLowConf =
    confidenceTier === "low" || confidenceTier === "unsupported" || condition === "uncertain";

  const statusConfig = useMemo(() => {
    if (isLowConf) {
      return {
        headline: "INCONCLUSIVE / LOW CONFIDENCE",
        sub: "Recording ambiguity prevents definitive fault localization.",
        cardCls: "border-[#2B4336] bg-[#0B1510]/95 text-[#EAE6DF]",
        pillCls: "bg-[#13221A] text-[#B8C5BD] border-[#2A4033]",
        dotCls: "bg-[#7C8F83]",
        icon: <AlertTriangle size={18} className="text-[#8FA196] shrink-0" />,
      };
    }
    if (condition === "engine_knock") {
      return {
        headline: "ENGINE KNOCKING DETECTED",
        sub: "Orange-Red Warning · Pistons, Connecting Rods, Crankshaft & Oil Pan highlighted.",
        cardCls: "border-[#C86B32] bg-[#0D1611]/95 text-[#EAE6DF]",
        pillCls: "bg-[#2C150E] text-[#FDBA74] border-[#8C3B19]",
        dotCls: "bg-[#EA580C] animate-ping",
        icon: <ShieldAlert size={18} className="text-[#EA580C] shrink-0" />,
      };
    }
    if (condition === "misfire") {
      return {
        headline: "ENGINE MISFIRE DETECTED",
        sub: "Amber Warning · Spark Plugs, Ignition Coils (COP) & Fuel Injectors highlighted.",
        cardCls: "border-[#D97706] bg-[#0D1611]/95 text-[#EAE6DF]",
        pillCls: "bg-[#2A1B0E] text-[#FCD34D] border-[#784919]",
        dotCls: "bg-[#F59E0B] animate-ping",
        icon: <AlertTriangle size={18} className="text-[#F59E0B] shrink-0" />,
      };
    }
    return {
      headline: "ENGINE CONDITION HEALTHY",
      sub: "Normal 1-3-4-2 combustion cadence · Green baseline verification active.",
      cardCls: "border-[#3B6E4C] bg-[#0B1510]/95 text-[#EAE6DF]",
      pillCls: "bg-[#11291B] text-[#86EFAC] border-[#23633B]",
      dotCls: "bg-[#4ADE80]",
      icon: <CheckCircle2 size={18} className="text-[#4ADE80] shrink-0" />,
    };
  }, [condition, isLowConf]);

  const handleReplay = () => {
    setManualReplay(true);
    setBonnetOpen(true);
    setCutawayMode(true);
    setCameraView(null);
    setPlayToken((k) => k + 1);
  };

  const fallbackUi = (
    <div className="p-8 bg-[#0B140F] text-center space-y-4">
      <div className="mx-auto h-12 w-12 rounded-full bg-[#13231A] border border-[#284233] flex items-center justify-center text-[#D86C2B]">
        <Wrench size={20} />
      </div>
      <div className="space-y-1">
        <div className="text-[16px] font-semibold text-[#EAE6DF]">{statusConfig.headline}</div>
        <p className="text-[13px] text-[#8FA196] max-w-md mx-auto">
          3D WebGL preview is unavailable in this display environment. Prediction confidence:{" "}
          <strong className="font-mono-data text-[#F2E8D5]">{Math.round(confidence * 100)}%</strong>{" "}
          ({predictionToLabel(condition)}).
        </p>
      </div>
      {onInspectSubsystem && (
        <Button size="sm" variant="secondary" onClick={onInspectSubsystem}>
          Open Subsystem Inspection Breakdown
        </Button>
      )}
    </div>
  );

  return (
    <div className="rounded-[10px] border border-[#1C2E24] bg-[#0B140F] overflow-hidden shadow-[0_14px_36px_rgba(0,0,0,0.5)]">
      {/* Top Dark Tactical Control Strip */}
      <div className="px-4 py-3 bg-[#0D1812] border-b border-[#1C2E24] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 flex-wrap">
          <Badge tone={isLowConf ? "neutral" : condition === "healthy" ? "healthy" : condition === "engine_knock" ? "fault" : "warning"} dot>
            {PHASE_LABELS[phase]}
          </Badge>
          <span className="text-[11px] font-mono-data uppercase tracking-wider text-[#8FA196] hidden sm:inline">
            2.0L DOHC I4 · 2023 Toyota Corolla Diagnostic Bay
          </span>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Button type="button" size="sm" onClick={handleReplay}>
            <Play size={13} strokeWidth={1.75} />
            Replay Sequence
          </Button>

          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => {
              setManualReplay(false);
              setBonnetOpen((o) => !o);
            }}
          >
            <RotateCcw size={13} strokeWidth={1.5} />
            {bonnetOpen ? "Close Bonnet" : "Open Bonnet"}
          </Button>

          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => {
              setManualReplay(false);
              setCutawayMode((c) => !c);
            }}
          >
            <Eye size={13} strokeWidth={1.5} />
            {cutawayMode ? "Solid Cover" : "Cutaway Reveal"}
          </Button>
        </div>
      </div>

      {/* Interactive Architectural Diagnostic Bay Stage */}
      <div className={`relative w-full ${compact ? "h-[420px]" : "h-[500px]"} bg-[#08110C] select-none`}>
        {webglOk ? (
          <Scene3DErrorBoundary fallback={fallbackUi}>
            {!sceneReady && (
              <div className="absolute inset-0 z-20 bg-[#08110C] flex flex-col items-center justify-center gap-3 text-[#8FA196]">
                <Activity size={26} className="animate-spin text-[#D86C2B]" />
                <div className="text-[14px] font-semibold text-[#EAE6DF]">
                  Loading 2023 Toyota Corolla Diagnostic Bay...
                </div>
                <div className="text-[12px] font-mono-data text-[#7C8F83]">
                  Streaming Toyota Corolla &amp; 34-node engine GLB models
                </div>
              </div>
            )}
            <Canvas
              shadows
              dpr={[1, 1.5]}
              camera={{ position: [3.85, 1.68, 4.10], fov: 38, near: 0.1, far: 80 }}
            >
              <color attach="background" args={["#08110C"]} />
              <fog attach="fog" args={["#08110C", 10, 24]} />

              {/* Neutral Studio Automotive Lighting Rig */}
              <hemisphereLight args={["#FFFFFF", "#1B252E", 0.72]} />
              <directionalLight
                position={[4.5, 7.5, 5.5]}
                intensity={1.75}
                castShadow
                shadow-mapSize-width={1024}
                shadow-mapSize-height={1024}
              />
              <directionalLight position={[-5, 4, -4]} intensity={0.9} color="#E2E8F0" />
              <directionalLight position={[0, 3, -6]} intensity={0.65} color="#CBD5E1" />

              <Suspense fallback={null}>
                <DiagnosticBayScene
                  condition={condition}
                  confidence={confidence}
                  confidenceTier={confidenceTier}
                  playToken={playToken}
                  autoPlay={manualReplay}
                  bonnetOpenTarget={bonnetOpen}
                  cutawayMode={cutawayMode}
                  cameraViewPreset={cameraView}
                  controlsRef={controlsRef}
                  onPhaseChange={setPhase}
                  onWarningVisible={setWarningVisible}
                  onReady={() => setSceneReady(true)}
                />
              </Suspense>
            </Canvas>
          </Scene3DErrorBoundary>
        ) : (
          fallbackUi
        )}

        {/* Top-Right Compact Camera Presets */}
        <div className="absolute top-3 right-3 flex items-center gap-1 bg-[#0C1611]/95 border border-[#284033] rounded-full p-1 shadow-md">
          <Camera size={13} className="text-[#8FA196] mx-1.5" />
          {(
            [
              { id: "sedan", label: "Exterior" },
              { id: "front_bay", label: "Front Bay" },
              { id: "closeup", label: "Engine" },
            ] as const
          ).map((cam) => (
            <button
              key={cam.id}
              type="button"
              onClick={() => {
                setManualReplay(false);
                if (cam.id !== "sedan") setBonnetOpen(true);
                setCameraView(cam.id);
              }}
              className={`px-2.5 py-1 rounded-full text-[11px] font-mono-data uppercase tracking-wider font-semibold transition-colors ${
                cameraView === cam.id
                  ? "bg-[#C86B32] text-[#FFF5EC]"
                  : "text-[#B8C5BD] hover:bg-[#16281E] hover:text-white"
              }`}
            >
              {cam.label}
            </button>
          ))}
        </div>

        {/* Animated Prediction & Confidence Diagnostic Alert Overlay */}
        {warningVisible && (
          <div className="absolute bottom-3 left-3 right-3 sm:right-auto sm:max-w-lg pointer-events-none">
            <div
              className={`pointer-events-auto rounded-[10px] border px-4 py-3 shadow-[0_12px_32px_rgba(0,0,0,0.65)] flex items-center justify-between gap-4 transition-all duration-300 ${statusConfig.cardCls}`}
            >
              <div className="flex items-start gap-3 min-w-0">
                <div className="mt-0.5">{statusConfig.icon}</div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-mono-data font-bold tracking-wider ${statusConfig.pillCls}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${statusConfig.dotCls}`} />
                      {statusConfig.headline}
                    </span>
                  </div>
                  <p className="text-[12px] text-[#B8C5BD] mt-1 leading-snug">{statusConfig.sub}</p>
                </div>
              </div>

              <div className="text-right shrink-0 pl-3 border-l border-[#23382C]">
                <div className="font-mono-data text-[20px] font-bold text-[#D86C2B] leading-none">
                  {Math.round(confidence * 100)}%
                </div>
                <div className="text-[10px] font-mono-data text-[#8FA196] uppercase mt-1">
                  Score
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Crisp Footer Hint */}
      <div className="px-4 py-2.5 bg-[#0D1812] border-t border-[#1C2E24] flex flex-wrap items-center justify-between gap-2 text-[12px] text-[#7C8F83]">
        <span>Drag to orbit · Scroll to zoom · Toggle Bonnet or Cutaway Reveal anytime</span>
        <span className="font-mono-data text-[11px] text-[#D86C2B]">
          GSAP Sequence + Dual GLTF 2.0 Scene
        </span>
      </div>
    </div>
  );
}

useGLTF.preload("/models/autoguard_corolla.glb");
useGLTF.preload("/models/autoguard_i4_engine.glb");
