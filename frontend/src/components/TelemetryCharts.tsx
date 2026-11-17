import { useState } from "react";
import { Card } from "./ui";

export function SparklineChart({
  values,
  color = "#38BDF8",
  height = 32,
}: {
  values: number[];
  color?: string;
  height?: number;
}) {
  if (!values || values.length < 2) {
    return (
      <div className="flex items-end gap-0.5 h-7 opacity-40">
        {[35, 50, 42, 68, 60, 75, 65, 82].map((v, i) => (
          <div key={i} className="w-1 rounded-full bg-[#475569]" style={{ height: `${v}%` }} />
        ))}
      </div>
    );
  }

  const w = 120;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const span = Math.max(max - min, 0.01);

  const pts = values.map((v, idx) => {
    const x = (idx / (values.length - 1)) * w;
    const y = height - 4 - ((v - min) / span) * (height - 8);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const areaPts = `0,${height} ${pts.join(" ")} ${w},${height}`;

  return (
    <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} className="overflow-visible">
      <defs>
        <linearGradient id={`spark-dark-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.22" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={areaPts} fill={`url(#spark-dark-${color.replace("#", "")})`} />
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={pts.join(" ")}
      />
    </svg>
  );
}

export function CnnEpochChart({
  epochs,
}: {
  epochs: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_acc: number;
    time_sec: number;
  }>;
}) {
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  if (!epochs.length) return null;

  const w = 520;
  const h = 210;
  const padL = 44;
  const padR = 44;
  const padT = 18;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxLoss = Math.max(...epochs.flatMap((e) => [e.train_loss, e.val_loss]), 1.2);
  const xFor = (idx: number) =>
    padL + (epochs.length === 1 ? plotW / 2 : (idx / (epochs.length - 1)) * plotW);
  const yLoss = (loss: number) => padT + plotH - (loss / maxLoss) * plotH;
  const yAcc = (acc: number) => padT + plotH - acc * plotH;

  const trainPath = epochs.map((e, i) => `${xFor(i)},${yLoss(e.train_loss)}`).join(" ");
  const valPath = epochs.map((e, i) => `${xFor(i)},${yLoss(e.val_loss)}`).join(" ");
  const accPath = epochs.map((e, i) => `${xFor(i)},${yAcc(e.val_acc)}`).join(" ");

  const active = hoverEpoch != null ? epochs.find((e) => e.epoch === hoverEpoch) : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between flex-wrap gap-2 text-[12px]">
        <div className="flex items-center gap-4 text-[#CBD5E1]">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#38BDF8]" /> Train loss
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#F59E0B]" /> Validation loss
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#22C55E]" /> Validation accuracy
          </span>
        </div>
        <div className="font-mono-data text-[12px] text-[#94A3B8]">
          {active
            ? `Epoch ${active.epoch}: train=${active.train_loss.toFixed(3)} · val=${active.val_loss.toFixed(3)} · acc=${(active.val_acc * 100).toFixed(1)}%`
            : "Hover points to inspect epoch metrics"}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-52 rounded-[8px] bg-[#090D16] border border-[#1E293B]"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = padT + frac * plotH;
          const lossTick = ((1 - frac) * maxLoss).toFixed(2);
          const accTick = `${Math.round((1 - frac) * 100)}%`;
          return (
            <g key={frac}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeWidth="1" />
              <text x={padL - 6} y={y + 3} textAnchor="end" fill="#94A3B8" fontSize="10" fontFamily="JetBrains Mono">
                {lossTick}
              </text>
              <text x={w - padR + 6} y={y + 3} textAnchor="start" fill="#22C55E" fontSize="10" fontFamily="JetBrains Mono">
                {accTick}
              </text>
            </g>
          );
        })}

        <polyline fill="none" stroke="#38BDF8" strokeWidth="2.2" points={trainPath} />
        <polyline fill="none" stroke="#F59E0B" strokeWidth="2.2" points={valPath} />
        <polyline fill="none" stroke="#22C55E" strokeWidth="2" strokeDasharray="4 3" points={accPath} />

        {epochs.map((e, i) => {
          const x = xFor(i);
          return (
            <g
              key={e.epoch}
              onMouseEnter={() => setHoverEpoch(e.epoch)}
              onMouseLeave={() => setHoverEpoch(null)}
              className="cursor-pointer"
            >
              <line
                x1={x}
                y1={padT}
                x2={x}
                y2={h - padB}
                stroke={hoverEpoch === e.epoch ? "rgba(56, 189, 248, 0.16)" : "transparent"}
                strokeWidth="12"
              />
              <circle cx={x} cy={yLoss(e.train_loss)} r="4" fill="#38BDF8" stroke="#090D16" strokeWidth="1.5" />
              <circle cx={x} cy={yLoss(e.val_loss)} r="4" fill="#F59E0B" stroke="#090D16" strokeWidth="1.5" />
              <circle cx={x} cy={yAcc(e.val_acc)} r="4" fill="#22C55E" stroke="#090D16" strokeWidth="1.5" />
              <text x={x} y={h - 9} textAnchor="middle" fill="#94A3B8" fontSize="10" fontFamily="JetBrains Mono">
                Ep {e.epoch}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ConfusionMatrixCard({
  title,
  subtitle,
  classes,
  matrix,
}: {
  title: string;
  subtitle: string;
  classes: string[];
  matrix: number[][];
}) {
  const shortName = (c: string) =>
    c === "engine_knock" ? "Knock" : c === "healthy" ? "Healthy" : "Misfire";

  return (
    <Card flat className="p-5 flex flex-col gap-4">
      <div>
        <h4 className="text-[16px] font-semibold text-[#F8FAFC]">{title}</h4>
        <p className="text-[13px] text-[#94A3B8] mt-0.5">{subtitle}</p>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[260px]">
          <div className="grid grid-cols-[76px_repeat(3,1fr)] gap-1.5 text-[11px] font-medium text-[#94A3B8] mb-1.5">
            <div className="text-right pr-2">True \ Pred</div>
            {classes.map((c) => (
              <div key={c} className="text-center">
                {shortName(c)}
              </div>
            ))}
          </div>

          {matrix.map((row, rIdx) => {
            const rowSum = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <div key={rIdx} className="grid grid-cols-[76px_repeat(3,1fr)] gap-1.5 mb-1.5">
                <div className="flex items-center justify-end pr-2 text-[12px] font-medium text-[#CBD5E1]">
                  {shortName(classes[rIdx] || "")}
                </div>
                {row.map((val, cIdx) => {
                  const ratio = val / rowSum;
                  const isDiag = rIdx === cIdx;
                  const cls = isDiag
                    ? "bg-[#22C55E]/15 border-[#22C55E]/35 text-[#4ADE80]"
                    : val > 0
                      ? "bg-[#EF4444]/15 border-[#EF4444]/35 text-[#F87171]"
                      : "bg-[#090D16] border-[#1E293B] text-[#64748B]";
                  return (
                    <div
                      key={cIdx}
                      className={`rounded-[8px] border ${cls} py-2.5 px-2 text-center font-mono-data`}
                    >
                      <div className="text-[15px] font-semibold">{val}</div>
                      <div className="text-[11px] opacity-80">{Math.round(ratio * 100)}%</div>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
