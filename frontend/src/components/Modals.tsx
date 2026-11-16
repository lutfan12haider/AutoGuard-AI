import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import type { InferenceMode } from "../types/diagnostic";

export function AnalyzingOverlay({ mode: _mode }: { mode: InferenceMode }) {
  return (
    <div className="fixed inset-0 z-50 bg-[#FAFAF9]/85 backdrop-blur-sm flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="w-full max-w-md studio-card p-8 text-center"
      >
        <div className="h-14 w-14 mx-auto mb-5 rounded-full bg-[#EEF2FF] border border-[#C7D2FE] flex items-center justify-center text-[#4F46E5]">
          <Activity size={24} strokeWidth={1.5} className="animate-pulse" />
        </div>

        <h3 className="text-[20px] font-semibold text-[#0F172A]">Checking your engine</h3>
        <p className="text-[14px] text-[#64748B] mt-2">
          Listening to the recording for signs of knock or misfire.
        </p>

        <div className="mt-6 h-1.5 w-full rounded-full bg-[#E7E5E4] overflow-hidden">
          <motion.div
            className="h-full bg-[#4F46E5] rounded-full"
            initial={{ x: "-100%", width: "40%" }}
            animate={{ x: "260%" }}
            transition={{ repeat: Infinity, duration: 1.25, ease: "easeInOut" }}
          />
        </div>
      </motion.div>
    </div>
  );
}
