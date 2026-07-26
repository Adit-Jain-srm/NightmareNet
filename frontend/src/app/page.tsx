import Navbar from "@/components/Navbar";
import NeuralBackground from "@/components/NeuralBackground";
import Hero from "@/components/Hero";
import GuidedDemo from "@/components/GuidedDemo";
import Architecture from "@/components/Architecture";
import QuickStart from "@/components/QuickStart";
import Playground from "@/components/Playground";
import ResilienceLab from "@/components/ResilienceLab";
import TrainingLab from "@/components/TrainingLab";
import PipelineLab from "@/components/PipelineLab";
import FileUpload from "@/components/FileUpload";
import ModelViewer from "@/components/ModelViewer";
import Status from "@/components/Status";
import Footer from "@/components/Footer";
import ScrollNavigator from "@/components/ScrollNavigator";

export default function Home() {
  return (
    <>
      <NeuralBackground />
      <Navbar />
      <main className="relative z-10">
        {/* 1. Hero — instant first impression */}
        <section id="hero">
          <Hero />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 2. Guided Demo — the "aha moment" */}
        <section id="demo">
          <GuidedDemo />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 3. Architecture — how the 4 phases work */}
        <section id="architecture">
          <Architecture />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 4. Quick Start — get started in 3 lines */}
        <section id="quickstart">
          <QuickStart />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 5. Full Playground — deeper exploration */}
        <div className="neural-grid">
          <section id="playground">
            <Playground />
          </section>

          <div className="section-glow-line max-w-5xl mx-auto" />

          {/* 6. Resilience Lab — measure robustness */}
          <section id="resilience">
            <ResilienceLab />
          </section>
        </div>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 7. Training Lab — advanced configuration */}
        <div className="mesh-gradient">
          <section id="training">
            <TrainingLab />
          </section>
        </div>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 8. Pipeline Lab — end-to-end training */}
        <section id="pipeline">
          <PipelineLab />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 9. Advanced tools */}
        <section id="upload">
          <FileUpload />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        <section id="viewer">
          <ModelViewer />
        </section>

        <div className="section-glow-line max-w-5xl mx-auto" />

        {/* 9. System status */}
        <section id="status">
          <Status />
        </section>
      </main>
      <ScrollNavigator />
      <Footer />
    </>
  );
}
