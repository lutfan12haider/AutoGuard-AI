import { useState } from "react";
import type { EngineCondition } from "../types/diagnostic";
import { Tabs } from "./ui";

type SubsystemKey = "cylinders" | "ignition" | "knock_sensor" | "belt";

const SUBSYSTEM_INFO: Record<
  SubsystemKey,
  {
    title: string;
    healthyNote: string;
    knockNote: string;
    misfireNote: string;
  }
> = {
  cylinders: {
    title: "Cylinders & combustion",
    healthyNote: "All four cylinders sound balanced with no supported fault pattern.",
    knockNote: "Check engine oil level, fuel octane quality, and internal component wear.",
    misfireNote: "Irregular combustion cadence may occur when a cylinder misfires.",
  },
  ignition: {
    title: "Spark plugs & ignition coils",
    healthyNote: "Firing rhythm appears steady and consistent with the healthy class.",
    knockNote: "Inspect ignition timing and combustion systems if knocking persists.",
    misfireNote: "Inspect spark plugs and check ignition coils for wear or defects.",
  },
  knock_sensor: {
    title: "Combustion & knock sensor",
    healthyNote: "No metallic knocking or pinging pattern detected on the engine block.",
    knockNote: "Verify fuel quality and manufacturer octane requirements; inspect combustion systems.",
    misfireNote: "Check for engine warning lights and diagnostic trouble codes.",
  },
  belt: {
    title: "Lubrication & fuel delivery",
    healthyNote: "Continue routine vehicle maintenance and monitor unusual sounds.",
    knockNote: "Check engine oil level and condition; top up or replace using correct specification.",
    misfireNote: "Inspect fuel injectors and check for fuel delivery issues.",
  },
};

export function EngineSchematic({
  condition = "healthy",
  compact = false,
}: {
  condition?: EngineCondition | "idle";
  compact?: boolean;
}) {
  const [selected, setSelected] = useState<SubsystemKey>("cylinders");

  const isKnock = condition === "engine_knock";
  const isMisfire = condition === "misfire";
  const isHealthy = condition === "healthy" || condition === "idle";

  const info = SUBSYSTEM_INFO[selected];
  const zoneStatusText = isKnock
    ? info.knockNote
    : isMisfire
      ? info.misfireNote
      : info.healthyNote;

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-[12px] border border-[#1E293B] bg-[#090D16] p-4">
        <div className="flex items-center justify-between text-[12px] text-[#94A3B8] mb-2">
          <span>Engine schematic</span>
          <span>Click a subsystem to inspect</span>
        </div>

        <svg
          viewBox="0 0 560 220"
          className={`w-full ${compact ? "h-44" : "h-52"} select-none`}
          role="img"
          aria-label="Interactive engine diagram"
        >
          {/* Belt & Pulley Assembly */}
          <g
            onClick={() => setSelected("belt")}
            onMouseEnter={() => setSelected("belt")}
            className="cursor-pointer"
          >
            <rect
              x="28"
              y="62"
              width="76"
              height="116"
              rx="8"
              fill="#111827"
              stroke={selected === "belt" ? "#38BDF8" : isMisfire ? "#EF4444" : "#334155"}
              strokeWidth={selected === "belt" || isMisfire ? 1.8 : 1}
            />
            <path
              d="M52 78 L80 78 L90 154 L42 154 Z"
              fill="none"
              stroke={isMisfire ? "#EF4444" : "#64748B"}
              strokeWidth="3.5"
              strokeLinejoin="round"
            />
            <circle cx="66" cy="84" r="13" fill="#090D16" stroke="#64748B" strokeWidth="1.5" />
            <circle cx="66" cy="84" r="3.5" fill="#38BDF8" />
            <circle cx="52" cy="148" r="10" fill="#090D16" stroke="#64748B" strokeWidth="1.5" />
            <circle cx="82" cy="146" r="8" fill="#090D16" stroke={isMisfire ? "#EF4444" : "#64748B"} strokeWidth="1.5" />
            <text x="66" y="196" textAnchor="middle" fill="#94A3B8" fontSize="11" fontFamily="Inter">
              Oil & Fuel
            </text>
          </g>

          {/* Engine Block Housing */}
          <g
            onClick={() => setSelected("cylinders")}
            onMouseEnter={() => setSelected("cylinders")}
            className="cursor-pointer"
          >
            <rect
              x="122"
              y="52"
              width="326"
              height="134"
              rx="8"
              fill="#111827"
              stroke={selected === "cylinders" ? "#38BDF8" : "#334155"}
              strokeWidth={selected === "cylinders" ? 1.8 : 1.2}
            />
            <rect x="132" y="28" width="306" height="24" rx="4" fill="#1E293B" stroke="#334155" strokeWidth="1" />
            <path d="M140 186 L430 186 L414 204 L156 204 Z" fill="#0F172A" stroke="#1E293B" strokeWidth="1" />
          </g>

          {/* 4 Cylinders + Pistons + Spark Coils */}
          {[0, 1, 2, 3].map((idx) => {
            const cx = 168 + idx * 78;
            const pistonY = idx === 0 || idx === 3 ? 88 : 112;
            const cylFault = (isKnock && idx === 2) || (isMisfire && idx === 1);
            const cylColor = cylFault
              ? isKnock
                ? "#F59E0B"
                : "#EF4444"
              : isHealthy
                ? "#22C55E"
                : "#38BDF8";

            return (
              <g key={idx}>
                <rect
                  x={cx - 24}
                  y="64"
                  width="48"
                  height="94"
                  rx="4"
                  fill="#090D16"
                  stroke={cylFault ? cylColor : "#1E293B"}
                  strokeWidth={cylFault ? 1.6 : 1}
                />

                <ellipse
                  cx={cx}
                  cy={pistonY - 10}
                  rx="16"
                  ry="7"
                  fill={cylColor}
                  opacity={cylFault ? 0.4 : 0.18}
                >
                  <animate
                    attributeName="opacity"
                    values={cylFault ? "0.2;0.55;0.2" : "0.1;0.26;0.1"}
                    dur={isMisfire && idx === 1 ? "0.8s" : `${1.2 + idx * 0.25}s`}
                    repeatCount="indefinite"
                  />
                </ellipse>

                <rect
                  x={cx - 19}
                  y={pistonY}
                  width="38"
                  height="24"
                  rx="3"
                  fill="#1E293B"
                  stroke={cylFault ? cylColor : "#64748B"}
                  strokeWidth="1.2"
                />
                <line x1={cx - 16} y1={pistonY + 6} x2={cx + 16} y2={pistonY + 6} stroke="#64748B" strokeWidth="1" />
                <line x1={cx} y1={pistonY + 24} x2={cx} y2="172" stroke="#64748B" strokeWidth="3.5" />
                <circle cx={cx} cy="172" r="5.5" fill="#090D16" stroke="#94A3B8" strokeWidth="1.5" />

                <g
                  onClick={() => setSelected("ignition")}
                  onMouseEnter={() => setSelected("ignition")}
                  className="cursor-pointer"
                >
                  <rect
                    x={cx - 8}
                    y="15"
                    width="16"
                    height="22"
                    rx="3"
                    fill="#090D16"
                    stroke={isMisfire && idx === 1 ? "#EF4444" : selected === "ignition" ? "#38BDF8" : "#475569"}
                    strokeWidth="1.4"
                  />
                  <line
                    x1={cx}
                    y1="37"
                    x2={cx}
                    y2="64"
                    stroke={isMisfire && idx === 1 ? "#EF4444" : "#38BDF8"}
                    strokeWidth="1.8"
                  />
                  <circle cx={cx} cy="22" r="2.5" fill={isMisfire && idx === 1 ? "#EF4444" : "#38BDF8"} />
                </g>

                <text x={cx} y="198" textAnchor="middle" fill="#94A3B8" fontSize="10" fontFamily="JetBrains Mono">
                  {idx + 1}
                </text>
              </g>
            );
          })}

          {/* Knock Sensor */}
          <g
            onClick={() => setSelected("knock_sensor")}
            onMouseEnter={() => setSelected("knock_sensor")}
            className="cursor-pointer"
          >
            <rect
              x="466"
              y="74"
              width="70"
              height="86"
              rx="8"
              fill="#111827"
              stroke={isKnock ? "#F59E0B" : selected === "knock_sensor" ? "#38BDF8" : "#334155"}
              strokeWidth={isKnock || selected === "knock_sensor" ? 1.8 : 1}
            />
            <line x1="448" y1="116" x2="466" y2="116" stroke={isKnock ? "#F59E0B" : "#38BDF8"} strokeWidth="2" />
            <circle cx="501" cy="110" r="13" fill="none" stroke={isKnock ? "#F59E0B" : "#38BDF8"} strokeWidth="1.4" strokeDasharray="3 2" />
            {isKnock && (
              <circle cx="501" cy="110" r="20" fill="none" stroke="#F59E0B" strokeWidth="1.2" opacity="0.5">
                <animate attributeName="r" values="12;24" dur="1.2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.7;0" dur="1.2s" repeatCount="indefinite" />
              </circle>
            )}
            <circle cx="501" cy="110" r="4.5" fill={isKnock ? "#F59E0B" : "#38BDF8"} />
            <text x="501" y="144" textAnchor="middle" fill="#94A3B8" fontSize="10" fontFamily="Inter">
              Sensor
            </text>
          </g>
        </svg>
      </div>

      {/* Clean Subsystem Summary */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="text-[14px] font-semibold text-[#F8FAFC]">{info.title}</div>
          <p className="text-[13px] text-[#94A3B8] mt-0.5">{zoneStatusText}</p>
        </div>

        <Tabs
          value={selected}
          onChange={setSelected}
          layoutId="schematic-zone-tab"
          items={[
            { id: "cylinders", label: "Cylinders" },
            { id: "ignition", label: "Ignition" },
            { id: "knock_sensor", label: "Combustion" },
            { id: "belt", label: "Oil & Fuel" },
          ]}
        />
      </div>
    </div>
  );
}
