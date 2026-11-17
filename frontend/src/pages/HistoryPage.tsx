import { useMemo, useState } from "react";
import { ChevronRight, Download, FileAudio, History, Plus, Search, Trash2 } from "lucide-react";
import type { AppPage, DiagnosticRecord, StatsResponse } from "../types/diagnostic";
import {
  exportHistoryToCsv,
  formatDuration,
  formatTimestamp,
  predictionToLabel,
} from "../services/api";
import { StatusBadge } from "../components/StatusBadge";
import {
  Button,
  Card,
  EmptyState,
  Skeleton,
  Stat,
  Table,
  TableHeader,
  TableRow,
} from "../components/ui";

type SortKey = "newest" | "confidence_desc" | "duration_desc";

export function HistoryPage({
  records,
  stats,
  loading,
  onNavigate,
  onInspectRecord,
  onDeleteRecord,
}: {
  records: DiagnosticRecord[];
  stats: StatsResponse | null;
  loading: boolean;
  onNavigate: (p: AppPage) => void;
  onInspectRecord: (rec: DiagnosticRecord) => void;
  onDeleteRecord: (scanId: number) => Promise<void>;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortKey>("newest");
  const [page, setPage] = useState(1);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const PAGE_SIZE = 8;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = records.filter((r) => {
      const label = predictionToLabel(r.prediction).toLowerCase();
      const matchesQ = !q || r.filename.toLowerCase().includes(q) || label.includes(q);
      const matchesF = filter === "all" || r.prediction === filter;
      return matchesQ && matchesF;
    });

    return [...list].sort((a, b) => {
      if (sortBy === "confidence_desc") return b.confidence - a.confidence;
      if (sortBy === "duration_desc") return b.duration_sec - a.duration_sec;
      return b.id - a.id;
    });
  }, [records, query, filter, sortBy]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageItems = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-[28px] font-semibold text-[#F8FAFC] leading-tight">History</h2>
          <p className="text-[14px] text-[#94A3B8] mt-1">
            Review past engine checks, inspect repair recommendations, or export your log.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={filtered.length === 0}
            onClick={() => exportHistoryToCsv(filtered)}
          >
            <Download size={14} strokeWidth={1.5} /> Download CSV
          </Button>
          <Button type="button" onClick={() => onNavigate("diagnosis")}>
            <Plus size={15} strokeWidth={1.5} /> Check engine
          </Button>
        </div>
      </div>

      {/* Clean 3-Stat Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Stat label="Total checks" value={(stats?.total_scans ?? records.length).toLocaleString()} />
        <Stat label="Healthy engines" value={(stats?.healthy_results ?? 0).toLocaleString()} />
        <Stat label="Fault patterns found" value={(stats?.detected_faults ?? 0).toLocaleString()} />
      </div>

      <Card className="overflow-hidden">
        <div className="p-4 border-b border-[#1E293B] flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <div className="flex items-center gap-2 rounded-[8px] bg-[#090D16] border border-[#1E293B] px-3 py-2 flex-1 focus-within:border-[#38BDF8]">
            <Search size={15} strokeWidth={1.5} className="text-[#94A3B8]" />
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              placeholder="Search recordings..."
              className="bg-transparent text-[13px] text-[#F8FAFC] placeholder-[#64748B] w-full outline-none"
            />
          </div>

          <select
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setPage(1);
            }}
            aria-label="Filter by result"
            className="rounded-[8px] bg-[#090D16] border border-[#1E293B] px-3 py-2 text-[13px] text-[#F8FAFC] outline-none"
          >
            <option value="all">All results</option>
            <option value="healthy">Healthy</option>
            <option value="engine_knock">Engine Knocking</option>
            <option value="misfire">Misfire</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortKey)}
            aria-label="Sort recordings"
            className="rounded-[8px] bg-[#090D16] border border-[#1E293B] px-3 py-2 text-[13px] text-[#F8FAFC] outline-none"
          >
            <option value="newest">Newest first</option>
            <option value="confidence_desc">Highest confidence</option>
            <option value="duration_desc">Longest recording</option>
          </select>
        </div>

        {loading ? (
          <div className="p-6 space-y-3">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : pageItems.length === 0 ? (
          <EmptyState
            icon={<History size={20} strokeWidth={1.5} />}
            title={records.length === 0 ? "No history yet" : "No matching recordings"}
            description={
              records.length === 0
                ? "Your engine sound checks will be saved here automatically."
                : "Try clearing your search or switching the filter."
            }
            action={
              records.length === 0 ? (
                <Button size="sm" onClick={() => onNavigate("diagnosis")}>
                  Check engine
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    setQuery("");
                    setFilter("all");
                  }}
                >
                  Clear filters
                </Button>
              )
            }
          />
        ) : (
          <Table>
            <TableHeader>
              <th className="py-3 px-4">Recording</th>
              <th className="py-3 px-4">Prediction</th>
              <th className="py-3 px-4">Confidence</th>
              <th className="py-3 px-4">Length</th>
              <th className="py-3 px-4 text-right">Actions</th>
            </TableHeader>
            <tbody>
              {pageItems.map((item) => {
                const { date, time } = formatTimestamp(item.created_at);
                const confPct = Math.round(item.confidence * 100);
                return (
                  <TableRow key={item.id}>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2.5">
                        <FileAudio size={16} strokeWidth={1.5} className="text-[#38BDF8] shrink-0" />
                        <div className="min-w-0">
                          <div className="text-[13px] font-medium text-[#F8FAFC] truncate max-w-[240px]">
                            {item.filename}
                          </div>
                          <div className="text-[12px] text-[#94A3B8]">
                            {date} · {time}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <StatusBadge prediction={item.prediction} shortLabel />
                    </td>
                    <td className="py-3.5 px-4 font-mono-data text-[13px] font-semibold text-[#F8FAFC]">
                      {confPct}%
                    </td>
                    <td className="py-3.5 px-4 font-mono-data text-[12px] text-[#94A3B8]">
                      {formatDuration(item.duration_sec)}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <div className="inline-flex items-center gap-1.5">
                        <Button
                          type="button"
                          variant="secondary"
                          size="sm"
                          onClick={() => onInspectRecord(item)}
                        >
                          View <ChevronRight size={13} strokeWidth={1.5} />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={deletingId === item.id}
                          onClick={async () => {
                            setDeletingId(item.id);
                            try {
                              await onDeleteRecord(item.id);
                            } finally {
                              setDeletingId(null);
                            }
                          }}
                          aria-label={`Delete ${item.filename}`}
                        >
                          <Trash2 size={14} strokeWidth={1.5} className="text-[#94A3B8] hover:text-[#F87171]" />
                        </Button>
                      </div>
                    </td>
                  </TableRow>
                );
              })}
            </tbody>
          </Table>
        )}

        {filtered.length > PAGE_SIZE && (
          <div className="px-4 py-3 border-t border-[#1E293B] flex items-center justify-between text-[13px] text-[#94A3B8]">
            <span className="font-mono-data">
              {(safePage - 1) * PAGE_SIZE + 1}–{Math.min(safePage * PAGE_SIZE, filtered.length)} of{" "}
              {filtered.length}
            </span>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={safePage <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={safePage >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
