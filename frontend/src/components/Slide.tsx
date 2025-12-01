"use client";

import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface SlideProps {
    title: string;
    children: React.ReactNode;
    onNext?: () => void;
    onPrev?: () => void;
    isFirst?: boolean;
    isLast?: boolean;
}

export default function Slide({ title, children, onNext, onPrev, isFirst, isLast }: SlideProps) {
    return (
        <motion.div
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            className="flex flex-col h-full w-full max-w-6xl mx-auto p-12"
        >
            <header className="mb-12 border-b border-zinc-800 pb-6">
                <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-violet-400">
                    {title}
                </h2>
            </header>

            <div className="flex-1 overflow-y-auto">
                {children}
            </div>

            <footer className="mt-12 flex items-center justify-between border-t border-zinc-800 pt-6">
                <button
                    onClick={onPrev}
                    disabled={isFirst}
                    className={`flex items-center gap-2 px-6 py-3 rounded-full transition-all ${isFirst
                        ? "opacity-0 cursor-default"
                        : "hover:bg-zinc-800 text-zinc-400 hover:text-white"
                        }`}
                >
                    <ChevronLeft className="w-5 h-5" />
                    Previous
                </button>

                <div className="flex gap-2">
                    {/* Progress indicators could go here */}
                </div>

                <button
                    onClick={onNext}
                    disabled={isLast}
                    className={`flex items-center gap-2 px-6 py-3 rounded-full transition-all ${isLast
                        ? "opacity-50 cursor-not-allowed"
                        : "bg-cyan-600 hover:bg-cyan-500 text-white"
                        }`}
                >
                    Next
                    <ChevronRight className="w-5 h-5" />
                </button>
            </footer>
        </motion.div>
    );
}
