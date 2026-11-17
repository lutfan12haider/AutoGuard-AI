import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import type { InferenceMode } from "../types/diagnostic";

export function AnalyzingOverlay({ mode: _mode }: { mode: InferenceMode }) {
  return (
    <div className="fixed inset-0 z-50 bg-[#090D16]/85 backdrop-blur-sm flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="w-full max-w-md studio-card p-8 text-center"
      >
        <div className="h-14 w-14 mx-auto mb-5 rounded-full bg-[#0EA5E9]/15 border border-[#38BDF8]/35 flex items-center justify-center text-[#38BDF8]">
          <Activity size={24} strokeWidth={1.5} className="animate-pulse" />
        </div>

        <h3 className="text-[20px] font-semibold text-[#F8FAFC]">Analyzing engine acoustics</h3>
        <p className="text-[14px] text-[#94A3B8] mt-2">
          Evaluating healthy, misfire, and engine knocking patterns and preparing repair recommendations.
        </p>

        <div className="mt-6 h-1.5 w-full rounded-full bg-[#1E293B] overflow-hidden">
          <motion.div
            className="h-full bg-[#38BDF8] rounded-full"
            initial={{ x: "-100%", width: "40%" }}
            animate={{ x: "260%" }}
            transition={{ repeat: Infinity, duration: 1.25, ease: "easeInOut" }}
          />
        </div>
      </motion.div>
    </div>
  );
}
