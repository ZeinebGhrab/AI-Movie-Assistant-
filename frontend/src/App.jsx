import { useState, useEffect, useCallback } from "react";

const API = "http://localhost:8000/api";

const GENRE_COLORS = {
  "sci-fi":  { bg: "#0C2340", border: "#378ADD", text: "#85B7EB" },
  romance:   { bg: "#2D0E1C", border: "#D4537E", text: "#ED93B1" },
  action:    { bg: "#2A0D05", border: "#D85A30", text: "#F0997B" },
  drama:     { bg: "#1A1A0A", border: "#639922", text: "#97C459" },
  thriller:  { bg: "#1E0E2A", border: "#7F77DD", text: "#AFA9EC" },
  default:   { bg: "#1A1A18", border: "#888780", text: "#B4B2A9" },
};

const MODE_META = {
  content:       { icon: "🎯", label: "Content-Based",    desc: "Based on movie attributes" },
  collaborative: { icon: "🤝", label: "Collaborative",    desc: "Users who liked what you liked" },
  hybrid:        { icon: "🔀", label: "Hybrid",           desc: "Best of both worlds" },
};

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 3, background: "#2C2C2A",
        borderRadius: 2, overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: `hsl(${180 + pct * 0.6}, 70%, 55%)`,
          borderRadius: 2,
          transition: "width 0.6s ease",
        }} />
      </div>
      <span style={{ fontSize: 12, color: "#888780", minWidth: 36, textAlign: "right" }}>
        {(score).toFixed(3)}
      </span>
    </div>
  );
}

function GenrePill({ genre }) {
  const c = GENRE_COLORS[genre] || GENRE_COLORS.default;
  return (
    <span style={{
      fontSize: 11, padding: "2px 8px", borderRadius: 20,
      background: c.bg, border: `1px solid ${c.border}`,
      color: c.text, fontWeight: 500, letterSpacing: "0.04em",
      textTransform: "uppercase",
    }}>{genre}</span>
  );
}

function MovieCard({ movie, rank, delay = 0 }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  return (
    <div style={{
      background: "#111110",
      border: "0.5px solid #2C2C2A",
      borderRadius: 10,
      padding: "14px 16px",
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(12px)",
      transition: "opacity 0.4s ease, transform 0.4s ease",
      position: "relative",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", top: 0, left: 0,
        width: 3, height: "100%",
        background: `hsl(${180 + movie.score * 60}, 65%, 50%)`,
        borderRadius: "10px 0 0 10px",
      }} />
      <div style={{ paddingLeft: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{
              fontSize: 11, fontWeight: 600, color: "#444441",
              minWidth: 20, fontFamily: "monospace",
            }}>#{rank}</span>
            <span style={{ fontSize: 15, fontWeight: 500, color: "#D3D1C7" }}>
              {movie.title}
            </span>
          </div>
          <GenrePill genre={movie.genre} />
        </div>
        <p style={{
          fontSize: 12, color: "#5F5E5A", lineHeight: 1.5,
          margin: "0 0 10px 30px",
        }}>{movie.description}</p>
        <div style={{ paddingLeft: 30 }}>
          <ScoreBar score={movie.score} />
        </div>
      </div>
    </div>
  );
}

function StatPill({ label, value, accent }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      background: "#111110", border: "0.5px solid #2C2C2A",
      borderRadius: 8, padding: "10px 14px",
    }}>
      <span style={{ fontSize: 11, color: "#5F5E5A", marginBottom: 4 }}>{label}</span>
      <span style={{ fontSize: 18, fontWeight: 600, color: accent || "#D3D1C7" }}>
        {value}
      </span>
    </div>
  );
}

function UserCard({ user, selected, onClick }) {
  return (
    <button onClick={onClick} style={{
      background: selected ? "#1A1A18" : "transparent",
      border: `0.5px solid ${selected ? "#888780" : "#2C2C2A"}`,
      borderRadius: 8, padding: "10px 14px",
      cursor: "pointer", textAlign: "left", width: "100%",
      transition: "all 0.15s ease",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "#D3D1C7", fontWeight: selected ? 500 : 400 }}>
          User {user.user_id}
        </span>
        <span style={{ fontSize: 11, color: "#5F5E5A" }}>
          {user.n_ratings} ratings
        </span>
      </div>
      <div style={{ marginTop: 6, display: "flex", gap: 4, flexWrap: "wrap" }}>
        {Object.entries(user.liked_genres || {}).map(([g, count]) => (
          <span key={g} style={{
            fontSize: 10, padding: "1px 6px",
            background: (GENRE_COLORS[g] || GENRE_COLORS.default).bg,
            color: (GENRE_COLORS[g] || GENRE_COLORS.default).text,
            border: `0.5px solid ${(GENRE_COLORS[g] || GENRE_COLORS.default).border}`,
            borderRadius: 12,
          }}>{g} ×{count}</span>
        ))}
      </div>
    </button>
  );
}

function EvalCard({ name, metrics }) {
  const entries = Object.entries(metrics);
  return (
    <div style={{
      background: "#111110", border: "0.5px solid #2C2C2A",
      borderRadius: 10, padding: "14px 16px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 16 }}>{MODE_META[name]?.icon}</span>
        <span style={{ fontSize: 13, fontWeight: 500, color: "#B4B2A9" }}>
          {MODE_META[name]?.label}
        </span>
      </div>
      {entries.map(([k, v]) => {
        const pct = Math.round(v * 100);
        return (
          <div key={k} style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 11, color: "#5F5E5A" }}>{k}</span>
              <span style={{ fontSize: 12, color: "#B4B2A9", fontFamily: "monospace" }}>
                {v.toFixed(4)}
              </span>
            </div>
            <div style={{ height: 3, background: "#2C2C2A", borderRadius: 2 }}>
              <div style={{
                width: `${pct}%`, height: "100%", borderRadius: 2,
                background: pct > 50 ? "#1D9E75" : pct > 20 ? "#BA7517" : "#A32D2D",
                transition: "width 0.8s ease",
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("recommend");
  const [users, setUsers] = useState([]);
  const [genres, setGenres] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.5);
  const [genre, setGenre] = useState("");
  const [recs, setRecs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [evalData, setEvalData] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/users`).then(r => r.json()).then(setUsers).catch(() => {});
    fetch(`${API}/genres`).then(r => r.json()).then(setGenres).catch(() => {});
  }, []);

  const recommend = useCallback(async () => {
    if (!selectedUser) return;
    setLoading(true);
    setError(null);
    setRecs(null);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: selectedUser.user_id,
          mode, top_k: topK,
          alpha,
          genre: genre || null,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      setRecs(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [selectedUser, mode, topK, alpha, genre]);

  const runEval = useCallback(async () => {
    setEvalLoading(true);
    try {
      const res = await fetch(`${API}/evaluate?top_k=${topK}`);
      setEvalData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setEvalLoading(false);
    }
  }, [topK]);

  const TAB_STYLE = (active) => ({
    padding: "8px 18px", borderRadius: 6, fontSize: 13,
    fontWeight: active ? 500 : 400, cursor: "pointer", border: "none",
    background: active ? "#1A1A18" : "transparent",
    color: active ? "#D3D1C7" : "#5F5E5A",
    transition: "all 0.15s ease",
  });

  return (
    <div style={{
      minHeight: "100vh", background: "#0D0D0C",
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      color: "#D3D1C7",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        borderBottom: "0.5px solid #2C2C2A",
        padding: "0 32px",
        display: "flex", alignItems: "center", gap: 32,
        height: 52,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: "#1D9E75", display: "flex",
            alignItems: "center", justifyContent: "center",
            fontSize: 14,
          }}>▶</div>
          <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: "0.05em", color: "#D3D1C7" }}>
            VEC<span style={{ color: "#1D9E75" }}>REC</span>SYS
          </span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {["recommend", "evaluate"].map(t => (
            <button key={t} style={TAB_STYLE(tab === t)} onClick={() => setTab(t)}>
              {t === "recommend" ? "⚡ Recommend" : "📊 Evaluate"}
            </button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", fontSize: 11, color: "#444441" }}>
          Qdrant + MongoDB · all-MiniLM-L6-v2
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 52px)" }}>

        {/* Sidebar */}
        <div style={{
          width: 220, borderRight: "0.5px solid #2C2C2A",
          padding: "20px 14px", overflowY: "auto",
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          <p style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em", margin: "0 0 8px 2px" }}>
            SELECT USER
          </p>
          {users.map(u => (
            <UserCard
              key={u.user_id}
              user={u}
              selected={selectedUser?.user_id === u.user_id}
              onClick={() => { setSelectedUser(u); setRecs(null); }}
            />
          ))}
        </div>

        {/* Main */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>

          {tab === "recommend" && (
            <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 24 }}>

              {/* Controls */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <p style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em", margin: "0 0 8px" }}>
                    ALGORITHM
                  </p>
                  {Object.entries(MODE_META).map(([m, meta]) => (
                    <button key={m} onClick={() => setMode(m)} style={{
                      display: "block", width: "100%", textAlign: "left",
                      padding: "9px 12px", marginBottom: 4, borderRadius: 7,
                      background: mode === m ? "#1A1A18" : "transparent",
                      border: `0.5px solid ${mode === m ? "#5F5E5A" : "#1A1A18"}`,
                      cursor: "pointer", transition: "all 0.15s",
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span>{meta.icon}</span>
                        <div>
                          <div style={{ fontSize: 12, color: mode === m ? "#D3D1C7" : "#888780", fontWeight: mode === m ? 500 : 400 }}>
                            {meta.label}
                          </div>
                          <div style={{ fontSize: 10, color: "#444441", marginTop: 1 }}>{meta.desc}</div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {mode === "hybrid" && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em" }}>α BLEND</span>
                      <span style={{ fontSize: 11, color: "#1D9E75", fontFamily: "monospace" }}>
                        {alpha.toFixed(2)}
                      </span>
                    </div>
                    <input type="range" min="0" max="1" step="0.05"
                      value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))}
                      style={{ width: "100%", accentColor: "#1D9E75" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#444441", marginTop: 4 }}>
                      <span>content</span><span>collaborative</span>
                    </div>
                  </div>
                )}

                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em" }}>TOP K</span>
                    <span style={{ fontSize: 11, color: "#1D9E75" }}>{topK}</span>
                  </div>
                  <input type="range" min="1" max="10" step="1"
                    value={topK} onChange={e => setTopK(parseInt(e.target.value))}
                    style={{ width: "100%", accentColor: "#1D9E75" }} />
                </div>

                <div>
                  <p style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em", margin: "0 0 6px" }}>
                    GENRE FILTER
                  </p>
                  <select value={genre} onChange={e => setGenre(e.target.value)} style={{
                    width: "100%", background: "#111110",
                    border: "0.5px solid #2C2C2A", borderRadius: 6,
                    color: "#888780", padding: "7px 10px", fontSize: 12,
                    appearance: "none", cursor: "pointer",
                  }}>
                    <option value="">All genres</option>
                    {genres.map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                </div>

                <button
                  onClick={recommend}
                  disabled={!selectedUser || loading}
                  style={{
                    padding: "11px", borderRadius: 7, fontSize: 13,
                    fontWeight: 500, cursor: selectedUser ? "pointer" : "not-allowed",
                    background: selectedUser ? "#0F6E56" : "#111110",
                    color: selectedUser ? "#9FE1CB" : "#444441",
                    border: `0.5px solid ${selectedUser ? "#1D9E75" : "#2C2C2A"}`,
                    transition: "all 0.2s",
                    fontFamily: "inherit",
                  }}>
                  {loading ? "⏳ Searching vectors…" : "⚡ Get Recommendations"}
                </button>

                {error && (
                  <div style={{
                    padding: 10, borderRadius: 6, fontSize: 12,
                    background: "#1C0A0A", border: "0.5px solid #A32D2D",
                    color: "#F09595",
                  }}>{error}</div>
                )}
              </div>

              {/* Results */}
              <div>
                {!selectedUser && (
                  <div style={{
                    display: "flex", flexDirection: "column",
                    alignItems: "center", justifyContent: "center",
                    height: 300, color: "#444441", gap: 8,
                  }}>
                    <div style={{ fontSize: 32 }}>◈</div>
                    <span style={{ fontSize: 13 }}>Select a user to begin</span>
                  </div>
                )}

                {recs && (
                  <>
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(3, 1fr)",
                      gap: 10, marginBottom: 20,
                    }}>
                      <StatPill label="user" value={`#${recs.user_id}`} accent="#1D9E75" />
                      <StatPill label="mode" value={MODE_META[recs.mode]?.icon + " " + recs.mode} />
                      <StatPill label="latency" value={`${recs.latency_ms.toFixed(1)} ms`} accent="#BA7517" />
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {recs.recommendations.map((movie, i) => (
                        <MovieCard
                          key={movie.movie_id}
                          movie={movie}
                          rank={i + 1}
                          delay={i * 80}
                        />
                      ))}
                    </div>

                    {recs.recommendations.length === 0 && (
                      <div style={{
                        textAlign: "center", padding: 40,
                        color: "#444441", fontSize: 13,
                      }}>No recommendations found for this configuration.</div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {tab === "evaluate" && (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 10, color: "#444441", letterSpacing: "0.1em" }}>TOP K</span>
                    <span style={{ fontSize: 11, color: "#1D9E75" }}>{topK}</span>
                  </div>
                  <input type="range" min="1" max="10" step="1"
                    value={topK} onChange={e => setTopK(parseInt(e.target.value))}
                    style={{ width: "100%", accentColor: "#1D9E75" }} />
                </div>
                <button onClick={runEval} disabled={evalLoading} style={{
                  padding: "10px 20px", borderRadius: 7, fontSize: 13,
                  fontWeight: 500, cursor: "pointer", fontFamily: "inherit",
                  background: "#0C2340", color: "#85B7EB",
                  border: "0.5px solid #378ADD",
                }}>
                  {evalLoading ? "⏳ Evaluating…" : "📊 Run Evaluation"}
                </button>
              </div>

              {evalData && (
                <>
                  <p style={{ fontSize: 11, color: "#444441", marginBottom: 16 }}>
                    Leave-one-out evaluation · top_k = {evalData.top_k}
                  </p>
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                    gap: 14,
                  }}>
                    {Object.entries(evalData.results).map(([name, metrics]) => (
                      <EvalCard key={name} name={name} metrics={metrics} />
                    ))}
                  </div>

                  <div style={{
                    marginTop: 24, padding: "14px 16px",
                    background: "#111110", border: "0.5px solid #2C2C2A",
                    borderRadius: 10,
                  }}>
                    <p style={{ fontSize: 11, color: "#5F5E5A", margin: "0 0 10px", letterSpacing: "0.05em" }}>
                      METRIC REFERENCE
                    </p>
                    {[
                      ["Precision@K", "Of the K items recommended, what fraction were actually relevant?"],
                      ["Recall@K", "Of all items the user liked, what fraction appeared in the top K?"],
                      ["NDCG@K", "Ranking quality — relevant item at rank 1 scores higher than rank 5."],
                    ].map(([m, d]) => (
                      <div key={m} style={{
                        display: "flex", gap: 12, paddingBottom: 8,
                        borderBottom: "0.5px solid #1A1A18", marginBottom: 8,
                      }}>
                        <span style={{ fontSize: 12, color: "#1D9E75", minWidth: 100, fontWeight: 500 }}>{m}</span>
                        <span style={{ fontSize: 12, color: "#5F5E5A" }}>{d}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {!evalData && !evalLoading && (
                <div style={{
                  display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center",
                  height: 280, color: "#444441", gap: 8,
                }}>
                  <div style={{ fontSize: 32 }}>◈</div>
                  <span style={{ fontSize: 13 }}>Press "Run Evaluation" to compare all three models</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}