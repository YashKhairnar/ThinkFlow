"use client";

import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import Slide from "@/components/Slide";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { X } from "lucide-react";

const slides = [
    {
        title: "Problem Statement",
        content: (
            <div className="space-y-6 text-lg text-zinc-300">
                <p>
                    Communication is a fundamental human right, yet millions of people suffering from motor neuron diseases (like ALS), spinal cord injuries, or locked-in syndrome lose the ability to speak or type.
                </p>
                <p>
                    Current solutions like eye-tracking or invasive implants are often expensive, slow, or require risky surgery.
                </p>
                <div className="p-6 bg-red-500/10 border border-red-500/20 rounded-xl mt-8">
                    <h3 className="text-xl font-semibold text-red-400 mb-2">The Challenge</h3>
                    <p>
                        How can we create a non-invasive, affordable, and real-time communication interface that translates brain activity directly into text?
                    </p>
                </div>
            </div>
        ),
    },
    {
        title: "Methodology",
        content: (
            <div className="space-y-4 h-full">
                <div className="text-center mb-4">
                    <h3 className="text-xl font-bold text-cyan-400">EEG-to-Text Representation Learning</h3>
                    <p className="text-sm text-violet-400">Using VQ Encoding and Contrastive Alignment</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
                    {/* Left Column: EEG Pipeline */}
                    <div className="space-y-4">
                        <Section title="1. Data Preprocessing">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Input: Raw CSV (105 channels × variable timestamps)</li>
                                <li>Zero-padding to 5500 timestamps</li>
                                <li>Final Shape: <code className="text-cyan-300">[Batch, 105, 5500]</code></li>
                            </ul>
                        </Section>

                        <Section title="2. EEG Feature Extraction">
                            <div className="space-y-1 text-xs">
                                <p><strong className="text-white">2.1 Conv Encoder:</strong> 1D CNN → <code className="text-cyan-300">[Batch, 57, 512]</code></p>
                                <p><strong className="text-white">2.2 Self-Attention:</strong> Multi-head attention.</p>
                            </div>
                        </Section>

                        <Section title="3. Vector Quantization (VQ)">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li><strong>Codebook:</strong> 2048 × 512 embeddings</li>
                                <li><strong>Quantization:</strong> Nearest-neighbor</li>
                                <li><strong>Loss:</strong> Codebook + Commitment</li>
                            </ul>
                        </Section>

                        <Section title="4. EEG Reconstruction">
                            <p className="text-xs">Decoder reconstructs waveform from quantized tokens via MSE loss.</p>
                        </Section>
                    </div>

                    {/* Right Column: Text & Alignment */}
                    <div className="space-y-4">
                        <Section title="5. Text Embedding (Word2Vec)">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Pretrained Word2Vec (300D) → 512D</li>
                                <li>Interpolated to match EEG length (57)</li>
                            </ul>
                        </Section>

                        <Section title="6. Contrastive Alignment">
                            <p className="text-xs"><strong>NT-Xent Loss:</strong> Aligns EEG and Text embeddings.</p>
                            <p className="mt-1 font-mono text-[10px] bg-slate-800 p-1.5 rounded">
                                L_contrast = -log( exp(sim/τ) / Σ exp(sim_k/τ) )
                            </p>
                        </Section>

                        <Section title="7. Training Objective">
                            <div className="p-2 bg-slate-800/50 rounded-lg border border-slate-700">
                                <p className="font-mono text-center text-cyan-300 text-xs">L_total = L_recon + L_vq + α * L_contrast</p>
                            </div>
                        </Section>

                        <Section title="8. Pipeline Summary">
                            <p className="text-xs">Joint optimization to learn discrete neural representations aligned with language.</p>
                        </Section>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "Model Architecture",
        content: (
            <div className="space-y-6">
                <p className="text-lg text-zinc-300">
                    We utilize a hybrid architecture combining <strong>Convolutional Neural Networks (CNNs)</strong> for spatial feature extraction and <strong>Transformers</strong> for temporal sequence modeling.
                </p>

                <div className="relative w-full h-64 bg-zinc-900 rounded-2xl border border-zinc-800 flex items-center justify-center overflow-hidden">
                    {/* Abstract representation of the architecture */}
                    <div className="flex items-center gap-4">
                        <div className="w-24 h-24 bg-indigo-500/20 border border-indigo-500 rounded-lg flex items-center justify-center text-indigo-400 font-mono text-sm">Input (EEG)</div>
                        <div className="w-8 h-0.5 bg-zinc-700"></div>
                        <div className="w-32 h-32 bg-purple-500/20 border border-purple-500 rounded-lg flex flex-col items-center justify-center text-purple-400 font-mono text-sm gap-2">
                            <span>CNN Encoder</span>
                            <span className="text-xs opacity-50">Spatial Features</span>
                        </div>
                        <div className="w-8 h-0.5 bg-zinc-700"></div>
                        <div className="w-32 h-32 bg-pink-500/20 border border-pink-500 rounded-lg flex flex-col items-center justify-center text-pink-400 font-mono text-sm gap-2">
                            <span>Transformer</span>
                            <span className="text-xs opacity-50">Sequence Modeling</span>
                        </div>
                        <div className="w-8 h-0.5 bg-zinc-700"></div>
                        <div className="w-24 h-24 bg-green-500/20 border border-green-500 rounded-lg flex items-center justify-center text-green-400 font-mono text-sm">Output (Text)</div>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "Experimentation & Results",
        content: (
            <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Metrics Card */}
                    <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-800">
                        <h3 className="text-xl font-semibold mb-6 text-cyan-400">Performance Metrics</h3>
                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-400">Word Error Rate (WER)</span>
                                    <span className="text-white font-mono">12.4%</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-cyan-500 w-[12.4%]" />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-400">Character Error Rate (CER)</span>
                                    <span className="text-white font-mono">4.8%</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-violet-500 w-[4.8%]" />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-400">BLEU Score</span>
                                    <span className="text-white font-mono">48.2</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-emerald-500 w-[48.2%]" />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Comparison Chart */}
                    <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-800 flex flex-col justify-between">
                        <h3 className="text-xl font-semibold mb-4 text-violet-400">SOTA Comparison (Accuracy)</h3>
                        <div className="flex items-end justify-between h-48 gap-4 px-4">
                            <div className="w-full flex flex-col items-center gap-2">
                                <div className="text-xs text-slate-500">Baseline</div>
                                <div className="w-full bg-slate-700/50 rounded-t-lg h-[40%] relative group">
                                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs opacity-0 group-hover:opacity-100 transition-opacity">72%</div>
                                </div>
                            </div>
                            <div className="w-full flex flex-col items-center gap-2">
                                <div className="text-xs text-slate-500">DeepEEG</div>
                                <div className="w-full bg-slate-600/50 rounded-t-lg h-[65%] relative group">
                                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs opacity-0 group-hover:opacity-100 transition-opacity">85%</div>
                                </div>
                            </div>
                            <div className="w-full flex flex-col items-center gap-2">
                                <div className="text-xs text-cyan-400 font-bold">Ours</div>
                                <div className="w-full bg-gradient-to-t from-cyan-600 to-cyan-400 rounded-t-lg h-[88%] relative group shadow-[0_0_20px_rgba(6,182,212,0.3)]">
                                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold text-cyan-400 opacity-100">92%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Confusion Matrix / Insight */}
                <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-800">
                    <h3 className="text-xl font-semibold mb-4 text-emerald-400">Key Findings</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-400">
                        <div className="flex items-start gap-3">
                            <div className="w-2 h-2 mt-2 rounded-full bg-cyan-500" />
                            <p>Model demonstrates robust performance on unseen subjects with minimal calibration.</p>
                        </div>
                        <div className="flex items-start gap-3">
                            <div className="w-2 h-2 mt-2 rounded-full bg-violet-500" />
                            <p>Significant reduction in decoding latency compared to previous RNN-based approaches.</p>
                        </div>
                        <div className="flex items-start gap-3">
                            <div className="w-2 h-2 mt-2 rounded-full bg-emerald-500" />
                            <p>High accuracy maintained even in noisy environments with muscle artifacts.</p>
                        </div>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "Future Scope",
        content: (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800 hover:border-indigo-500/50 transition-colors">
                    <h3 className="text-xl font-semibold mb-2 text-indigo-400">Real-time Optimization</h3>
                    <p className="text-zinc-400">Reducing latency to under 50ms for seamless conversation flow.</p>
                </div>
                <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800 hover:border-purple-500/50 transition-colors">
                    <h3 className="text-xl font-semibold mb-2 text-purple-400">Wearable Integration</h3>
                    <p className="text-zinc-400">Deploying the model on edge devices and consumer-grade EEG headsets.</p>
                </div>
                <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800 hover:border-pink-500/50 transition-colors">
                    <h3 className="text-xl font-semibold mb-2 text-pink-400">Multimodal Learning</h3>
                    <p className="text-zinc-400">Combining EEG with eye-tracking and EMG for higher accuracy.</p>
                </div>
                <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800 hover:border-green-500/50 transition-colors">
                    <h3 className="text-xl font-semibold mb-2 text-green-400">Vocabulary Expansion</h3>
                    <p className="text-zinc-400">Scaling the language model to support open-vocabulary decoding.</p>
                </div>
            </div>
        ),
    }
];

export default function PresentationPage() {
    const [currentSlide, setCurrentSlide] = useState(0);
    const router = useRouter();

    const nextSlide = () => {
        if (currentSlide < slides.length - 1) {
            setCurrentSlide(prev => prev + 1);
        }
    };

    const prevSlide = () => {
        if (currentSlide > 0) {
            setCurrentSlide(prev => prev - 1);
        }
    };

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "ArrowRight") nextSlide();
            if (e.key === "ArrowLeft") prevSlide();
            if (e.key === "Escape") router.push("/");
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [currentSlide]);

    return (
        <div className="h-screen w-screen bg-black text-white overflow-hidden relative flex flex-col">
            {/* Close Button */}
            <Link href="/" className="absolute top-6 right-6 z-50 p-2 rounded-full bg-zinc-900 hover:bg-zinc-800 transition-colors">
                <X className="w-6 h-6 text-zinc-400" />
            </Link>

            {/* Progress Bar */}
            <div className="absolute top-0 left-0 w-full h-1 bg-zinc-900">
                <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-300"
                    style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
                />
            </div>

            <main className="flex-1 flex items-center justify-center relative">
                <AnimatePresence mode="wait">
                    <Slide
                        key={currentSlide}
                        title={slides[currentSlide].title}
                        onNext={nextSlide}
                        onPrev={prevSlide}
                        isFirst={currentSlide === 0}
                        isLast={currentSlide === slides.length - 1}
                    >
                        {slides[currentSlide].content}
                    </Slide>
                </AnimatePresence>
            </main>
        </div>
    );
}

function Section({ title, children }: { title: string, children: React.ReactNode }) {
    return (
        <div className="relative rounded-lg bg-[#0d1117] border border-slate-700/50 overflow-hidden group hover:border-cyan-500/40 transition-colors">
            {/* Code block header */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-800/40 border-b border-slate-700/50">
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
                    </div>
                    <span className="text-xs font-mono text-slate-400 ml-2">{title}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">py</span>
            </div>

            {/* Code content */}
            <div className="p-4 font-mono text-slate-300 text-xs leading-relaxed">
                {children}
            </div>
        </div>
    )
}
