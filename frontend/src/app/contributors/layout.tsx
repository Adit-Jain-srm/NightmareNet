import React from "react";

export default function ContributorsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-void text-text">
      <div className="pt-24">{children}</div>
    </div>
  );
}
