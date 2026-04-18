import { useState, useEffect, useRef } from 'react';
import GlassCard from '../components/GlassCard.jsx';
import { buildQuiz } from '../lib/quiz.js';

export default function QuizScreen({ words, count, mode, type, onFinish, onBack }) {
  const [questions] = useState(() => buildQuiz(words, count, mode, type));
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState([]); // { isCorrect, userAnswer }
  const [selected, setSelected] = useState(null); // for multiple choice
  const [input, setInput] = useState('');         // for typing
  const [revealed, setRevealed] = useState(false);
  const inputRef = useRef();

  const q = questions[current];
  const progress = (current / questions.length) * 100;

  useEffect(() => {
    if (q?.type === 'typing' && inputRef.current) {
      inputRef.current.focus();
    }
  }, [current, q]);

  if (questions.length === 0) {
    return (
      <div style={{ maxWidth: 540, margin: '0 auto', padding: '48px 20px', textAlign: 'center', position: 'relative', zIndex: 1 }}>
        <p style={{ color: 'rgba(255,255,255,0.5)' }}>단어가 부족해서 퀴즈를 만들 수 없어요 (최소 4개 필요)</p>
        <button onClick={onBack} style={{ marginTop: 20, color: '#6c8fff', background: 'none', fontSize: 14 }}>← 돌아가기</button>
      </div>
    );
  }

  function checkAnswer(answer) {
    const correct = q.answer.trim().toLowerCase();
    const user = answer.trim().toLowerCase();
    const isCorrect = q.type === 'typing'
      ? user === correct || correct.includes(user) || user.includes(correct)
      : user === correct;
    return isCorrect;
  }

  function submitAnswer(answer) {
    if (revealed) return;
    const isCorrect = checkAnswer(answer);
    if (q.type === 'multiple') setSelected(answer);
    setRevealed(true);
    setAnswers(prev => [...prev, { ...q, userAnswer: answer, isCorrect }]);
  }

  function next() {
    const nextIdx = current + 1;
    if (nextIdx >= questions.length) {
      onFinish([...answers]);
    } else {
      setCurrent(nextIdx);
      setSelected(null);
      setInput('');
      setRevealed(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') {
      if (!revealed && input.trim()) {
        submitAnswer(input);
      } else if (revealed) {
        next();
      }
    }
  }

  return (
    <div style={{ maxWidth: 540, margin: '0 auto', padding: '40px 20px 80px', position: 'relative', zIndex: 1 }}>
      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
        <button
          onClick={onBack}
          style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 999, color: 'rgba(255,255,255,0.6)', padding: '7px 14px', fontSize: 13, fontWeight: 600 }}
        >
          ← 나가기
        </button>
        <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 14, fontWeight: 600 }}>
          {current + 1} / {questions.length}
        </span>
        <div style={{ width: 72 }} />
      </div>

      {/* Progress bar */}
      <div style={{ height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2, marginBottom: 36, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${progress}%`,
          background: 'linear-gradient(90deg, #6c8fff, #a78bfa)',
          borderRadius: 2,
          transition: 'width 0.4s ease',
        }} />
      </div>

      {/* Question card */}
      <GlassCard
        key={current}
        style={{ padding: '32px 28px', marginBottom: 24, textAlign: 'center', animation: 'fadeUp 0.35s ease both' }}
        hover={false}
      >
        <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: 11, fontWeight: 600, letterSpacing: 1.2, textTransform: 'uppercase', marginBottom: 14 }}>
          {q.label}
        </p>
        <p style={{ fontSize: 34, fontWeight: 800, letterSpacing: -0.5, lineHeight: 1.2 }}>
          {q.question}
        </p>
        {q.type === 'typing' && (
          <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: 12, marginTop: 8 }}>
            Enter로 제출
          </p>
        )}
      </GlassCard>

      {/* Multiple choice */}
      {q.type === 'multiple' && (
        <div style={{ display: 'grid', gap: 10 }}>
          {q.choices.map((choice, i) => {
            const isSelected = selected === choice;
            const isCorrect = choice.trim().toLowerCase() === q.answer.trim().toLowerCase();
            let bg = 'rgba(255,255,255,0.06)';
            let border = '1px solid rgba(255,255,255,0.10)';
            let color = '#fff';
            if (revealed) {
              if (isCorrect) { bg = 'rgba(74,222,128,0.15)'; border = '1px solid rgba(74,222,128,0.4)'; color = '#4ade80'; }
              else if (isSelected && !isCorrect) { bg = 'rgba(248,113,113,0.13)'; border = '1px solid rgba(248,113,113,0.35)'; color = '#f87171'; }
            } else if (isSelected) {
              bg = 'rgba(108,143,255,0.15)'; border = '1px solid rgba(108,143,255,0.4)';
            }
            return (
              <button
                key={i}
                onClick={() => !revealed && submitAnswer(choice)}
                style={{
                  padding: '14px 18px', borderRadius: 14, fontSize: 15, fontWeight: 600,
                  background: bg, border, color,
                  textAlign: 'left', transition: 'all 0.15s',
                }}
              >
                <span style={{ color: 'rgba(255,255,255,0.3)', marginRight: 12, fontWeight: 700 }}>
                  {String.fromCharCode(65 + i)}
                </span>
                {choice}
                {revealed && isCorrect && <span style={{ float: 'right' }}>✓</span>}
                {revealed && isSelected && !isCorrect && <span style={{ float: 'right' }}>✗</span>}
              </button>
            );
          })}
        </div>
      )}

      {/* Typing input */}
      {q.type === 'typing' && (
        <div>
          <div style={{ display: 'flex', gap: 10 }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={revealed}
              placeholder="답을 입력하세요..."
              style={{
                flex: 1, padding: '14px 18px', borderRadius: 14, fontSize: 15, fontWeight: 600,
                background: revealed
                  ? checkAnswer(input)
                    ? 'rgba(74,222,128,0.12)'
                    : 'rgba(248,113,113,0.1)'
                  : 'rgba(255,255,255,0.07)',
                border: revealed
                  ? checkAnswer(input)
                    ? '1px solid rgba(74,222,128,0.4)'
                    : '1px solid rgba(248,113,113,0.35)'
                  : '1px solid rgba(255,255,255,0.12)',
                color: '#fff',
                transition: 'all 0.2s',
              }}
            />
            {!revealed && (
              <button
                onClick={() => input.trim() && submitAnswer(input)}
                style={{
                  padding: '14px 20px', borderRadius: 14, fontSize: 15, fontWeight: 700,
                  background: 'rgba(108,143,255,0.3)',
                  border: '1px solid rgba(108,143,255,0.4)',
                  color: '#fff', transition: 'opacity 0.15s',
                }}
              >
                →
              </button>
            )}
          </div>

          {revealed && (
            <div style={{ marginTop: 14, padding: '12px 16px', borderRadius: 12, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
              <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>정답</p>
              <p style={{ fontSize: 18, fontWeight: 700, color: '#4ade80' }}>{q.answer}</p>
            </div>
          )}
        </div>
      )}

      {/* Next button */}
      {revealed && (
        <button
          onClick={next}
          style={{
            width: '100%', marginTop: 24,
            padding: '15px', borderRadius: 14, fontSize: 15, fontWeight: 700,
            background: 'linear-gradient(135deg, rgba(108,143,255,0.55), rgba(167,139,250,0.45))',
            border: '1px solid rgba(108,143,255,0.35)',
            color: '#fff', animation: 'fadeUp 0.25s ease both',
            transition: 'opacity 0.15s, transform 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.opacity = '0.85'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          {current + 1 >= questions.length ? '결과 보기 →' : '다음 →'}
        </button>
      )}
    </div>
  );
}
