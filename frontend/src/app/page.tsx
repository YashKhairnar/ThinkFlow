import Link from "next/link";
import { BrainCircuit, ArrowRight, Activity, Zap, Cpu, Network } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#020617] overflow-hidden relative">

      {/* Neural Grid Background */}
      <div className="absolute inset-0 neural-grid opacity-30 pointer-events-none" />
      <div className="absolute inset-0 scanline opacity-20 pointer-events-none" />

      {/* Glowing Orbs */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-violet-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse" style={{ animationDelay: "2s" }} />

      <main className="relative z-10 max-w-6xl w-full px-6 text-center">
        <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white via-cyan-100 to-cyan-400/50 drop-shadow-[0_0_30px_rgba(6,182,212,0.3)]">
          ThinkFlow
        </h1>

        <p className="text-xl md:text-2xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed font-light">
          Decode raw <span className="text-cyan-400 font-medium">EEG signals</span> into fluent English using advanced <span className="text-violet-400 font-medium">Transformer models & LSTM</span>.
        </p>

        <div className="flex flex-col md:flex-row items-center justify-center gap-6">
          <Link href="/demo" className="group relative px-8 py-4 bg-cyan-500 text-black rounded-full font-bold text-lg hover:bg-cyan-400 transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:shadow-[0_0_40px_rgba(6,182,212,0.6)]">
            Try Demo
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>

          <Link href="/presentation" className="px-8 py-4 rounded-full glass-panel text-cyan-100 font-medium hover:bg-cyan-500/10 transition-colors border border-cyan-500/20 hover:border-cyan-500/50 flex items-center gap-2">
            <Network className="w-4 h-4" />
            View Presentation
          </Link>
        </div>

      </main>
    </div>
  );
}
