"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface WaveformProps {
  isActive?: boolean;
  color?: string;
}

export default function Waveform({ isActive = false, color = "#06b6d4" }: WaveformProps) {
  const [bars, setBars] = useState<number[]>(new Array(40).fill(10));

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
          className="w-1.5 rounded-full"
          style={{ backgroundColor: color, opacity: 0.8 }}
        />
      ))}
    </div>
  );
}
