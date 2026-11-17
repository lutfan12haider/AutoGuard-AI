# AutoGuard AI — 3D Engine Model & Visual Assets Documentation

## 1. 3D Engine Model (`/models/autoguard_i4_engine.glb`)
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
