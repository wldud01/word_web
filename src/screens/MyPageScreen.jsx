import { useState } from 'react';
import GlassCard from '../components/GlassCard.jsx';
import { loadWords, loadHistory, clearAll } from '../lib/storage.js';

export default function MyPageScreen({ onBack }) {
  const [tab, setTab] = useState('words'); // 'words' | 'history'
  const [words] = useState(() => loadWords());
  const [history] = useState(() => loadHistory());
  const [filter, setFilter] = useState('all'); // 'all' | 'weak' | 'strong'
  const [confirmClear, setConfirmClear] = useState(false);

  const totalQuizzes = history.length;
  const totalCorrect = history.reduce((s, h) => s + h.correct, 0);
  const totalAttempts = history.reduce((s, h) => s + h.total, 0);
  const overallScore = totalAttempts > 0 ? Math.round((totalCorrect / totalAttempts) * 100) : 0;

  const sorted = [...words].sort((a, b) => {
    const rateA = a.correct + a.wrong > 0 ? a.correct / (a.correct + a.wrong) : -1;
    const rateB = b.correct + b.wrong > 0 ? b.correct / (b.correct + b.wrong) : -1;
    return rateA - rateB;
  });

  const filteredWords = sorted.filter(w => {
    if (filter === 'weak') return w.wrong > w.correct;
    if (filter === 'strong') return w.correct > 0 && w.wrong === 0;
    return true;
  });

  function formatDate(iso) {
    const d = new Date(iso);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  }

  function handleClear() {
    clearAll();
    setConfirmClear(false);
    window.location.reload();
  }

  return (
    <div style={{ maxWidth: 580, margin: '0 auto', padding: '40px 20px 80px', position: 'relative', zIndex: 1 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 32 }}>
        <button
          onClick={onBack}
          style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 999, color: 'rgba(255,255,255,0.6)', padding: '7px 14px', fontSize: 13, fontWeight: 600 }}
        >
          ←
        </button>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800 }}>마이페이지</h1>
          <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: 12 }}>나의 학습 기록</p>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 28 }}>
        {[
          { label: '단어 수', value: words.length, color: '#6c8fff' },
          { label: '퀴즈 횟수', value: totalQuizzes, color: '#a78bfa' },
          { label: '전체 정답률', value: `${overallScore}%`, color: overallScore >= 70 ? '#4ade80' : '#fbbf24' },
        ].map(({ label, value, color }) => (
          <GlassCard key={label} style={{ padding: '16px 12px', textAlign: 'center' }} hover={false}>
            <p style={{ fontSize: 24, fontWeight: 800, color }}>{value}</p>
            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: 600, marginTop: 4 }}>{label}</p>
          </GlassCard>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {[{ v: 'words', l: '단어장' }, { v: 'history', l: '퀴즈 기록' }].map(({ v, l }) => (
          <button
            key={v}
            onClick={() => setTab(v)}
            style={{
              padding: '8px 20px', borderRadius: 10, fontSize: 13, fontWeight: 700,
              background: tab === v ? 'rgba(108,143,255,0.2)' : 'rgba(255,255,255,0.06)',
              border: tab === v ? '1px solid rgba(108,143,255,0.45)' : '1px solid rgba(255,255,255,0.1)',
              color: tab === v ? '#6c8fff' : 'rgba(255,255,255,0.5)',
              transition: 'all 0.15s',
            }}
          >
            {l}
          </button>
        ))}
      </div>

      {/* Word list tab */}
      {tab === 'words' && (
        <>
          {/* Filter */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            {[{ v: 'all', l: '전체' }, { v: 'weak', l: '🔴 취약' }, { v: 'strong', l: '🟢 완벽' }].map(({ v, l }) => (
              <button
                key={v}
                onClick={() => setFilter(v)}
                style={{
                  padding: '5px 12px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                  background: filter === v ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
                  border: filter === v ? '1px solid rgba(255,255,255,0.2)' : '1px solid rgba(255,255,255,0.08)',
                  color: filter === v ? '#fff' : 'rgba(255,255,255,0.4)',
                  transition: 'all 0.15s',
                }}
              >
                {l}
              </button>
            ))}
          </div>

          {filteredWords.length === 0 ? (
            <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: 14, textAlign: 'center', marginTop: 32 }}>
              {words.length === 0 ? '아직 단어가 없어요. 엑셀 파일을 업로드하세요.' : '해당 조건의 단어가 없어요.'}
            </p>
          ) : (
            <div style={{ display: 'grid', gap: 8 }}>
              {filteredWords.map((w, i) => {
                const total = w.correct + w.wrong;
                const rate = total > 0 ? Math.round((w.correct / total) * 100) : null;
                const barColor = rate === null ? 'rgba(255,255,255,0.2)' : rate >= 70 ? '#4ade80' : rate >= 40 ? '#fbbf24' : '#f87171';
                return (
                  <GlassCard key={i} style={{ padding: '12px 16px' }} hover={false}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <p style={{ fontWeight: 700, fontSize: 15 }}>{w.en}</p>
                          {rate !== null && (
                            <span style={{
                              fontSize: 11, fontWeight: 700, padding: '2px 7px', borderRadius: 6,
                              background: `${barColor}20`, color: barColor,
                            }}>
                              {rate}%
                            </span>
                          )}
                          {total === 0 && (
                            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.25)', fontWeight: 600 }}>미시도</span>
                          )}
                        </div>
                        <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, marginTop: 2 }}>{w.ko}</p>
                      </div>
                      <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 12 }}>
                        <span style={{ color: '#4ade80', fontSize: 12, fontWeight: 600 }}>{w.correct}✓</span>
                        <span style={{ color: 'rgba(255,255,255,0.2)', margin: '0 4px' }}>|</span>
                        <span style={{ color: '#f87171', fontSize: 12, fontWeight: 600 }}>{w.wrong}✗</span>
                      </div>
                    </div>
                    {total > 0 && (
                      <div style={{ marginTop: 8, height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${rate}%`, background: barColor, borderRadius: 2, transition: 'width 0.5s ease' }} />
                      </div>
                    )}
                  </GlassCard>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* History tab */}
      {tab === 'history' && (
        <div style={{ display: 'grid', gap: 10 }}>
          {history.length === 0 ? (
            <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: 14, textAlign: 'center', marginTop: 32 }}>
              아직 퀴즈 기록이 없어요.
            </p>
          ) : (
            history.map(h => {
              const pct = Math.round((h.correct / h.total) * 100);
              const color = pct >= 70 ? '#4ade80' : pct >= 50 ? '#fbbf24' : '#f87171';
              return (
                <GlassCard key={h.id} style={{ padding: '14px 18px' }} hover={false}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <p style={{ fontWeight: 700, fontSize: 15, marginBottom: 3 }}>
                        {h.correct}/{h.total} 정답
                      </p>
                      <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: 12 }}>{formatDate(h.date)}</p>
                    </div>
                    <div style={{
                      fontSize: 22, fontWeight: 800, color,
                    }}>
                      {pct}점
                    </div>
                  </div>
                </GlassCard>
              );
            })
          )}
        </div>
      )}

      {/* Clear data */}
      <div style={{ marginTop: 40, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 24 }}>
        {!confirmClear ? (
          <button
            onClick={() => setConfirmClear(true)}
            style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.2)', fontSize: 13, textDecoration: 'underline' }}
          >
            모든 데이터 초기화
          </button>
        ) : (
          <GlassCard style={{ padding: '16px 20px', border: '1px solid rgba(248,113,113,0.25)' }} hover={false}>
            <p style={{ color: '#f87171', fontSize: 14, fontWeight: 600, marginBottom: 12 }}>정말 모든 데이터를 삭제할까요?</p>
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                onClick={handleClear}
                style={{ padding: '8px 20px', borderRadius: 10, background: 'rgba(248,113,113,0.2)', border: '1px solid rgba(248,113,113,0.35)', color: '#f87171', fontSize: 13, fontWeight: 700 }}
              >
                삭제
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                style={{ padding: '8px 20px', borderRadius: 10, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.5)', fontSize: 13, fontWeight: 600 }}
              >
                취소
              </button>
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  );
}
