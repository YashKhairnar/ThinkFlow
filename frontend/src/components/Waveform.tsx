"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface WaveformProps {
  isActive: boolean;
  color?: "cyan" | "violet" | "orange";
}

export default function Waveform({ isActive, color = "cyan" }: WaveformProps) {
  const [bars, setBars] = useState<number[]>(new Array(40).fill(10));
  const barColor = color === "violet" ? "bg-violet-500" : color === "orange" ? "bg-orange-500" : "bg-cyan-500";

  useEffect(() => {
    if (!isActive) {
      setBars(new Array(40).fill(10));
      return;
    }

    const interval = setInterval(() => {
      setBars(prev => prev.map(() => Math.random() * 40 + 10));
    }, 100);

    return () => clearInterval(interval);
  }, [isActive]);

  return (
    <div className="flex items-center justify-center gap-1 h-24 w-full overflow-hidden">
      {bars.map((height, i) => (
        <motion.div
          key={i}
          initial={{ height: 10 }}
          animate={{ height: isActive ? height : 10 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className={`w-1 rounded-full ${barColor} transition-all duration-300 ease-in-out`}
        />
      ))}
    </div>
  );
}
