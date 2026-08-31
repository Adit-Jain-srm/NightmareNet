"use client";

import { useEffect, useState } from "react";

type Contributor = {
  login: string;
  avatar_url: string;
  html_url: string;
  contributions: number;
  prs?: number;
  issues?: number;
  last_active?: string;
};

export default function ContributorsPage() {
  const [data, setData] = useState<Contributor[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetch("/contributors.json")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load contributors: ${r.status}`);
        return r.json();
      })
      .then((json) => {
        if (mounted) setData(json as Contributor[]);
      })
      .catch((e) => {
        if (mounted) setError(String(e));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <main className="max-w-7xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-semibold mb-4">Contributors</h1>
      <p className="text-slate-400 mb-8">Thanks to everyone who contributes to this project.</p>

      {loading && <p>Loading contributors…</p>}
      {error && <p className="text-red-500">{error}</p>}

      {!loading && !error && data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {data.map((c) => (
            <a
              key={c.login}
              href={c.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-4 p-4 rounded-lg bg-white/[0.02] border border-white/[0.03] hover:bg-white/[0.04]"
            >
              <img src={c.avatar_url} alt={c.login} className="w-12 h-12 rounded-full" />
              <div>
                <div className="font-medium">{c.login}</div>
                <div className="text-sm text-slate-400">{c.contributions} contributions</div>
              </div>
              <div className="ml-auto text-right text-sm text-slate-400">
                <div>PRs: {c.prs ?? "—"}</div>
                <div>Issues: {c.issues ?? "—"}</div>
              </div>
            </a>
          ))}
        </div>
      )}

      {!loading && !error && (!data || data.length === 0) && (
        <p>No contributor data available.</p>
      )}
    </main>
  );
}
