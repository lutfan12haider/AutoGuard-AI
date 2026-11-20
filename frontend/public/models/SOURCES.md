# AutoGuard AI — 3D Engine & Vehicle Model Assets Documentation

## 1. 2023 Toyota Corolla Hybrid Articulated Vehicle Model (`/models/autoguard_corolla.glb`)
- **Format**: Binary glTF 2.0 (`.glb`)
- **Base Asset**: ["2023 Toyota Corolla Hybrid"](https://sketchfab.com/3d-models/2023-toyota-corolla-hybrid-cd2f6b34664442ad906a40bd00136881) by [**tonielpro520**](https://sketchfab.com/tonielpro520) (sourced via open-source WebGL release `matiasduclosmd-1778/vita-machines`)
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)** — http://creativecommons.org/licenses/by/4.0/
- **Pipeline Modifications (`scripts/prepare_corolla_glb.py`)**:
  - Decompressed & dequantized to standard IEEE `float32` / `uint32` WebGL-safe glTF 2.0.
  - Tuned body paint (`Carro_Pintura`) in Toyota Classic Silver Metallic clearcoat.
  - Sliced the sculpted Corolla bonnet panel along natural cowl/fender/headlight shut lines and parented it (plus a dark inner underliner pad) under `Bonnet_Hood_Pivot` at `[0.0, 0.962, 1.340]` for realistic animated bonnet articulation.
  - Built a matte black structural Corolla engine compartment tub (`Corolla_Engine_Bay_Tub`: firewall bulkhead, radiator core support & cooling fan, strut towers, intake/battery enclosures) housing `/models/autoguard_i4_engine.glb`.

*(Legacy fallback prototype `scripts/generate_sedan_glb.py` / `autoguard_sports_sedan.glb` superseded by `autoguard_corolla.glb`.)*

## 2. 3D Engine Model (`/models/autoguard_i4_engine.glb`)
- **Format**: Binary glTF 2.0 (`.glb`)
- **Generator**: `scripts/generate_engine_glb.py` (AutoGuard AI Procedural CAD GLB Builder v2.0)
- **License**: **CC0 1.0 Universal (Public Domain Dedication)** — generated in-repo from parametric CAD geometry specifically for AutoGuard AI.
- **Scene Graph Nodes (34 Individually Named & Selectable Components)**:
  - **Outer Structure & Lubrication**:
    - `Engine_Block` — 4-cylinder inline block with bored sleeves, deck plates, cooling ribs, and main bearing webs
    - `Cylinder_Head` — DOHC 16-valve upper head casting with spark plug wells
    - `Valve_Cover` — Ribbed upper cam cover with oil filler cap
    - `Oil_Pan` — Deep gunmetal lubrication sump and oil drain assembly
    - `Exhaust_Manifold` — 4-into-1 tubular exhaust header
  - **Ignition & Fuel Delivery (`misfire` Possible Inspection Areas)**:
    - `Spark_Plug_1`, `Spark_Plug_2`, `Spark_Plug_3`, `Spark_Plug_4`
    - `Ignition_Coil_1`, `Ignition_Coil_2`, `Ignition_Coil_3`, `Ignition_Coil_4`
    - `Fuel_Injector_1`, `Fuel_Injector_2`, `Fuel_Injector_3`, `Fuel_Injector_4`
    - `Fuel_Rail`
  - **Combustion & Rotating Assembly (`engine_knock` Possible Inspection Areas)**:
    - `Combustion_Chamber_1`, `Combustion_Chamber_2`, `Combustion_Chamber_3`, `Combustion_Chamber_4`
    - `Piston_1`, `Piston_2`, `Piston_3`, `Piston_4`
    - `Connecting_Rod_1`, `Connecting_Rod_2`, `Connecting_Rod_3`, `Connecting_Rod_4`
    - `Crankshaft`
  - **Valvetrain & Timing**:
    - `Camshaft_Intake`, `Camshaft_Exhaust`
    - `Intake_Valves`, `Exhaust_Valves`
    - `Timing_Drive_Belt`
