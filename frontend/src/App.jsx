import { useState, useEffect, useRef } from "react";
import "./index.css";

const API = "http://localhost:8080";
const PLACEHOLDER = "https://via.placeholder.com/300x450/0f0f14/00d4aa?text=No+Poster";

export default function App() {
  const [query, setQuery]       = useState("");
  const [suggestions, setSug]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [results, setResults]   = useState(null);
  const [posters, setPosters]   = useState({});
  const [meta, setMeta]         = useState(null);
  const [error, setError]       = useState("");
  const [stage, setStage]       = useState(null);
  const debounce                = useRef(null);

  // Autocomplete
  useEffect(() => {
    if (query.length < 2) { setSug([]); return; }
    clearTimeout(debounce.current);
    debounce.current = setTimeout(async () => {
      try {
        const r = await fetch(`${API}/search?q=${encodeURIComponent(query)}`);
        const d = await r.json();
        setSug(d.results || []);
      } catch { setSug([]); }
    }, 300);
  }, [query]);

  // Fetch posters after results load
  useEffect(() => {
    if (!results) return;
    const fetchPosters = async () => {
      try {
        const titles = results.map(r => r.title);
        const res    = await fetch(`${API}/movie-details`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ titles }),
        });
        const data = await res.json();
        setPosters(data);
      } catch { setPosters({}); }
    };
    fetchPosters();
  }, [results]);

  const handleSubmit = async (title = query) => {
    if (!title.trim()) return;
    setLoading(true);
    setError("");
    setResults(null);
    setPosters({});
    setMeta(null);
    setSug([]);

    setStage("retrieving");
    await new Promise(r => setTimeout(r, 800));
    setStage("ranking");

    try {
      const res  = await fetch(`${API}/recommend`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ title, n: 15 }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Something went wrong."); return; }
      setResults(data.recommendations);
      setMeta({ query: data.query, pool: data.candidates_pool });
      setQuery(data.query);
    } catch {
      setError("Cannot reach API. Is Flask running on port 8080?");
    } finally {
      setLoading(false);
      setStage(null);
    }
  };

  const pickSuggestion = (t) => { setQuery(t); setSug([]); handleSubmit(t); };

  return (
    <div style={styles.app}>

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logoWrap}>
          <span style={styles.logoIcon}>◈</span>
          <h1 style={styles.logo}>Film<span style={styles.logoAccent}>Finder</span></h1>
        </div>
        <p style={styles.tagline}>LightGCN Graph Retrieval · XGBoost Re-ranking</p>
        <p style={styles.tagline2}>Bollywood · South Indian · Korean Cinema</p>
      </header>

      {/* Pipeline */}
      <div style={styles.pipeline}>
        <PipeStep num="01" label="LightGCN Retrieval"
          desc="Graph embeddings → top-50 candidates"
          active={stage === "retrieving"} done={!!results} />
        <span style={styles.pipeArrow}>→</span>
        <PipeStep num="02" label="XGBoost Re-ranking"
          desc="Feature ranking → final top-15"
          active={stage === "ranking"} done={!!results} />
      </div>

      {/* Search */}
      <div style={styles.searchWrap}>
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            placeholder="Search Bollywood, Korean, South Indian films..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSubmit()}
          />
          <button style={{...styles.btn, opacity: loading ? 0.5 : 1}}
            onClick={() => handleSubmit()} disabled={loading}>
            {loading ? "..." : "Find →"}
          </button>
        </div>
        {suggestions.length > 0 && (
          <ul style={styles.suggestions}>
            {suggestions.map(s => (
              <li key={s} style={styles.suggestion}
                onClick={() => pickSuggestion(s)}
                onMouseEnter={e => e.target.style.color = "var(--teal)"}
                onMouseLeave={e => e.target.style.color = "var(--text)"}
              >{s}</li>
            ))}
          </ul>
        )}
      </div>

      {error && <div style={styles.error}>⚠ {error}</div>}

      {/* Meta */}
      {meta && (
        <div style={styles.meta}>
          <span>Query: <strong>{meta.query}</strong></span>
          <span>Pool: <strong>{meta.pool} candidates</strong></span>
          <span>Top <strong>15</strong> after XGBoost re-ranking</span>
        </div>
      )}

      {/* Results grid with posters */}
      {results && (
        <div style={styles.grid}>
          {results.map((r, i) => {
            const details = posters[r.title] || {};
            return (
              <div key={r.title} style={{
                ...styles.card,
                animationDelay: `${i * 50}ms`
              }}>
                {/* Poster */}
                <div style={styles.posterWrap}>
                  <img
                    src={details.poster || PLACEHOLDER}
                    alt={r.title}
                    style={styles.poster}
                    onError={e => { e.target.src = PLACEHOLDER; }}
                  />
                  <div style={styles.rankBadge}>#{r.rank}</div>
                  {details.language && (
                    <div style={styles.langBadge}>{details.language}</div>
                  )}
                </div>

                {/* Info */}
                <div style={styles.cardBody}>
                  <div style={styles.cardTitle}>{r.title}</div>

                  {details.overview && (
                    <div style={styles.overview}>
                      {details.overview.slice(0, 100)}...
                    </div>
                  )}

                  {/* Scores */}
                  <div style={styles.scores}>
                    <div style={styles.scoreRow}>
                      <span style={styles.scoreLabel}>GCN</span>
                      <div style={styles.scoreBarWrap}>
                        <div style={{
                          ...styles.scoreBar,
                          width: `${Math.round(r.gcn_score * 100)}%`,
                          background: "var(--teal)"
                        }} />
                      </div>
                      <span style={styles.scoreNum}>{r.gcn_score.toFixed(3)}</span>
                    </div>
                    <div style={styles.scoreRow}>
                      <span style={styles.scoreLabel}>XGB</span>
                      <div style={styles.scoreBarWrap}>
                        <div style={{
                          ...styles.scoreBar,
                          width: `${Math.round(Math.min(r.xgb_score / 5, 1) * 100)}%`,
                          background: "var(--gold)"
                        }} />
                      </div>
                      <span style={styles.scoreNum}>{r.xgb_score.toFixed(3)}</span>
                    </div>
                  </div>

                  {details.vote_avg && (
                    <div style={styles.voteAvg}>
                      ⭐ {details.vote_avg.toFixed(1)} / 10
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!results && !loading && !error && (
        <div style={styles.empty}>
          <div style={styles.emptyIcon}>◈</div>
          <p>Search for any Bollywood, South Indian or Korean movie</p>
          <div style={styles.exampleChips}>
            {["Parasite", "RRR", "Kantara", "3 Idiots", "Oldboy", "Pushpa"].map(m => (
              <span key={m} style={styles.chip}
                onClick={() => { setQuery(m); handleSubmit(m); }}>
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      <footer style={styles.footer}>
        FilmFinder · LightGCN + XGBoost · TMDB Dataset · React + Flask
      </footer>
    </div>
  );
}

function PipeStep({ num, label, desc, active, done }) {
  return (
    <div style={{
      ...styles.pipeStep,
      borderColor: active ? "var(--teal)" : done ? "rgba(0,212,170,0.3)" : "var(--border)",
      background:  active ? "rgba(0,212,170,0.06)" : "transparent",
    }}>
      <span style={styles.pipeNum}>{num}</span>
      <span style={styles.pipeLabel}>{label}</span>
      <span style={styles.pipeDesc}>{desc}</span>
    </div>
  );
}

const styles = {
  app:        { position:"relative", zIndex:1, maxWidth:1100, margin:"0 auto", padding:"0 24px 80px" },
  header:     { padding:"48px 0 28px", textAlign:"center" },
  logoWrap:   { display:"inline-flex", alignItems:"center", gap:10, marginBottom:8 },
  logoIcon:   { fontSize:28, color:"var(--teal)" },
  logo:       { fontFamily:"var(--syne)", fontSize:"clamp(32px,6vw,52px)", fontWeight:800, letterSpacing:"-0.04em" },
  logoAccent: { color:"var(--teal)" },
  tagline:    { fontSize:12, color:"var(--muted)", letterSpacing:"0.12em", textTransform:"uppercase" },
  tagline2:   { fontSize:11, color:"var(--teal)", letterSpacing:"0.08em", marginTop:4, opacity:0.7 },
  pipeline:   { display:"flex", alignItems:"center", gap:12, background:"var(--surface)", border:"1px solid var(--border)", borderRadius:10, padding:"14px 20px", marginBottom:24 },
  pipeStep:   { flex:1, border:"1px solid", borderRadius:6, padding:"10px 14px", transition:"all 0.3s" },
  pipeNum:    { display:"block", fontSize:10, color:"var(--teal)", letterSpacing:"0.1em", marginBottom:3 },
  pipeLabel:  { display:"block", fontSize:13, fontWeight:600, color:"var(--text)", marginBottom:2 },
  pipeDesc:   { display:"block", fontSize:11, color:"var(--muted)" },
  pipeArrow:  { fontSize:18, color:"var(--muted)", flexShrink:0 },
  searchWrap: { position:"relative", marginBottom:20 },
  inputRow:   { display:"flex", gap:8 },
  input:      { flex:1, background:"var(--surface)", border:"1px solid var(--border)", borderRadius:6, color:"var(--text)", fontFamily:"var(--mono)", fontSize:15, padding:"13px 16px", outline:"none" },
  btn:        { background:"var(--teal)", border:"none", borderRadius:6, color:"#000", cursor:"pointer", fontFamily:"var(--mono)", fontSize:13, fontWeight:600, padding:"13px 22px", whiteSpace:"nowrap" },
  suggestions:{ position:"absolute", top:"calc(100% + 4px)", left:0, right:0, background:"var(--surface)", border:"1px solid var(--border)", borderRadius:6, listStyle:"none", zIndex:20, maxHeight:240, overflowY:"auto", boxShadow:"0 12px 32px rgba(0,0,0,0.5)" },
  suggestion: { padding:"11px 16px", fontSize:13, cursor:"pointer", borderBottom:"1px solid var(--border)", color:"var(--text)", transition:"color 0.15s" },
  error:      { background:"rgba(255,71,87,0.07)", border:"1px solid rgba(255,71,87,0.3)", borderRadius:6, color:"#ff8090", fontSize:13, padding:"12px 16px", marginBottom:20 },
  meta:       { display:"flex", flexWrap:"wrap", gap:16, fontSize:12, color:"var(--muted)", marginBottom:20, padding:"10px 0", borderBottom:"1px solid var(--border)" },

  // Grid layout for cards
  grid:       { display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(200px, 1fr))", gap:16 },

  // Movie card
  card:       { background:"var(--surface)", border:"1px solid var(--border)", borderRadius:10, overflow:"hidden", transition:"transform 0.2s, border-color 0.2s", cursor:"default", animation:"fadeIn 0.4s ease both" },
  posterWrap: { position:"relative", width:"100%", aspectRatio:"2/3", overflow:"hidden" },
  poster:     { width:"100%", height:"100%", objectFit:"cover", display:"block", transition:"transform 0.3s" },
  rankBadge:  { position:"absolute", top:8, left:8, background:"rgba(0,0,0,0.8)", color:"var(--teal)", fontFamily:"var(--mono)", fontSize:11, fontWeight:700, padding:"3px 8px", borderRadius:4 },
  langBadge:  { position:"absolute", top:8, right:8, background:"rgba(0,212,170,0.15)", color:"var(--teal)", fontFamily:"var(--mono)", fontSize:9, padding:"3px 6px", borderRadius:4, border:"1px solid rgba(0,212,170,0.3)" },
  cardBody:   { padding:"12px" },
  cardTitle:  { fontSize:13, fontWeight:600, color:"var(--text)", marginBottom:6, lineHeight:1.3 },
  overview:   { fontSize:11, color:"var(--muted)", lineHeight:1.4, marginBottom:8 },
  scores:     { display:"flex", flexDirection:"column", gap:4, marginBottom:6 },
  scoreRow:   { display:"flex", alignItems:"center", gap:6 },
  scoreLabel: { fontSize:10, color:"var(--muted)", width:28, fontFamily:"var(--mono)" },
  scoreBarWrap:{ flex:1, height:3, background:"var(--border)", borderRadius:2, overflow:"hidden" },
  scoreBar:   { height:"100%", borderRadius:2, transition:"width 0.5s ease" },
  scoreNum:   { fontSize:10, color:"var(--muted)", fontFamily:"var(--mono)", width:36, textAlign:"right" },
  voteAvg:    { fontSize:11, color:"var(--gold)", marginTop:4 },

  // Empty state
  empty:      { textAlign:"center", padding:"60px 0", color:"var(--muted)" },
  emptyIcon:  { fontSize:36, color:"rgba(0,212,170,0.2)", marginBottom:14 },
  exampleChips:{ display:"flex", flexWrap:"wrap", gap:8, justifyContent:"center", marginTop:20 },
  chip:       { padding:"8px 16px", border:"1px solid var(--border)", borderRadius:20, fontSize:13, cursor:"pointer", color:"var(--muted)", transition:"all 0.2s" },

  footer:     { marginTop:60, textAlign:"center", fontSize:11, color:"var(--muted)", letterSpacing:"0.08em" },
};