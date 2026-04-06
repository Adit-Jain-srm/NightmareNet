import Navbar from "@/components/Navbar";
import NeuralBackground from "@/components/NeuralBackground";
import Hero from "@/components/Hero";
import GuidedDemo from "@/components/GuidedDemo";
import Architecture from "@/components/Architecture";
import QuickStart from "@/components/QuickStart";
import Playground from "@/components/Playground";
import ResilienceLab from "@/components/ResilienceLab";
import TrainingLab from "@/components/TrainingLab";
import FileUpload from "@/components/FileUpload";
import ModelViewer from "@/components/ModelViewer";
import Status from "@/components/Status";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <NeuralBackground />
      <Navbar />
      <main className="relative z-10">
        {/* 1. Hero — instant first impression */}
        <Hero />

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 2. Guided Demo — the "aha moment" */}
        <GuidedDemo />

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 3. Architecture — how the 4 phases work */}
        <Architecture />

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 4. Quick Start — get started in 3 lines */}
        <QuickStart />

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 5. Full Playground — deeper exploration */}
        <div className="neural-grid">
          <Playground />

          <div className="section-glow-line max-w-5xl mx-auto" />

          {/* 6. Resilience Lab — measure robustness */}
          <ResilienceLab />
        </div>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 7. Training Lab — advanced configuration */}
        <div className="mesh-gradient">
          <TrainingLab />
        </div>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 8. Advanced tools */}
        <FileUpload />

        <div className="section-glow-line max-w-5xl mx-auto" />

        <ModelViewer />

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 9. System status */}
        <Status />
      </main>
      <Footer />
    </>
  );
}
