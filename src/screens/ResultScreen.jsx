import GlassCard from '../components/GlassCard.jsx';

export default function ResultScreen({ results, onRetry, onHome, onMyPage }) {
  const total = results.length;
  const correct = results.filter(r => r.isCorrect).length;
  const wrong = total - correct;
  const score = Math.round((correct / total) * 100);

  const emoji = score >= 90 ? '🎉' : score >= 70 ? '😊' : score >= 50 ? '😐' : '😓';
  const message = score >= 90 ? '완벽해요!' : score >= 70 ? '잘했어요!' : score >= 50 ? '조금 더 노력해봐요' : '복습이 필요해요';

  return (
    <div style={{ maxWidth: 540, margin: '0 auto', padding: '48px 20px 80px', position: 'relative', zIndex: 1 }}>
      {/* Score card */}
      <GlassCard style={{ padding: '40px 28px', textAlign: 'center', marginBottom: 24, animation: 'fadeUp 0.4s ease both' }} hover={false}>
        <div style={{ fontSize: 52, marginBottom: 12 }}>{emoji}</div>
        <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 13, fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}>
          퀴즈 완료
        </p>
        <div style={{
          fontSize: 64, fontWeight: 800, letterSpacing: -2,
          background: score >= 70
            ? 'linear-gradient(135deg, #4ade80, #6c8fff)'
            : 'linear-gradient(135deg, #fbbf24, #f87171)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          marginBottom: 8,
        }}>
          {score}점
        </div>
        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
          {message}
        </p>
        <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
          <div style={{ background: 'rgba(74,222,128,0.12)', border: '1px solid rgba(74,222,128,0.25)', borderRadius: 12, padding: '10px 20px' }}>
            <p style={{ color: '#4ade80', fontSize: 22, fontWeight: 800 }}>{correct}</p>
            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: 600 }}>정답</p>
          </div>
          <div style={{ background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.22)', borderRadius: 12, padding: '10px 20px' }}>
            <p style={{ color: '#f87171', fontSize: 22, fontWeight: 800 }}>{wrong}</p>
            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: 600 }}>오답</p>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: '10px 20px' }}>
            <p style={{ color: '#fff', fontSize: 22, fontWeight: 800 }}>{total}</p>
            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: 600 }}>전체</p>
          </div>
        </div>
      </GlassCard>

      {/* Wrong answers */}
      {wrong > 0 && (
        <div style={{ marginBottom: 24 }}>
          <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11, fontWeight: 700, letterSpacing: 1.2, textTransform: 'uppercase', marginBottom: 12 }}>
            틀린 단어 복습
          </p>
          <div style={{ display: 'grid', gap: 8 }}>
            {results.filter(r => !r.isCorrect).map((r, i) => (
              <GlassCard
                key={i}
                style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid rgba(248,113,113,0.18)', background: 'rgba(248,113,113,0.05)' }}
                hover={false}
              >
                <div>
                  <p style={{ fontWeight: 700, fontSize: 15 }}>{r.word.en}</p>
                  <p style={{ color: '#4ade80', fontSize: 13, marginTop: 2 }}>{r.word.ko}</p>
                </div>
                {r.userAnswer && (
                  <div style={{ textAlign: 'right' }}>
                    <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: 11 }}>내 답</p>
                    <p style={{ color: '#f87171', fontSize: 13, fontWeight: 600 }}>{r.userAnswer}</p>
                  </div>
                )}
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {/* Correct answers */}
      {correct > 0 && (
        <div style={{ marginBottom: 28 }}>
          <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11, fontWeight: 700, letterSpacing: 1.2, textTransform: 'uppercase', marginBottom: 12 }}>
            맞은 단어
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {results.filter(r => r.isCorrect).map((r, i) => (
              <div
                key={i}
                style={{
                  background: 'rgba(74,222,128,0.1)', border: '1px solid rgba(74,222,128,0.2)',
                  borderRadius: 10, padding: '6px 12px',
                  fontSize: 13, fontWeight: 600, color: '#4ade80',
                }}
              >
                {r.word.en}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Buttons */}
      <div style={{ display: 'grid', gap: 10 }}>
        <button
          onClick={onRetry}
          style={{
            padding: '15px', borderRadius: 14, fontSize: 15, fontWeight: 700,
            background: 'linear-gradient(135deg, rgba(108,143,255,0.55), rgba(167,139,250,0.45))',
            border: '1px solid rgba(108,143,255,0.35)',
            color: '#fff', transition: 'opacity 0.15s, transform 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.opacity = '0.85'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          다시 풀기
        </button>
        <button
          onClick={onMyPage}
          style={{
            padding: '15px', borderRadius: 14, fontSize: 15, fontWeight: 600,
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
            color: 'rgba(255,255,255,0.7)', transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.10)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.06)'}
        >
          마이페이지 →
        </button>
        <button
          onClick={onHome}
          style={{
            padding: '12px', borderRadius: 14, fontSize: 14, fontWeight: 600,
            background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)',
          }}
        >
          홈으로
        </button>
      </div>
    </div>
  );
}
