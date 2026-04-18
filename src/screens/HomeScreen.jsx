import { useRef, useState } from 'react';
import ExcelJS from 'exceljs';
import GlassCard from '../components/GlassCard.jsx';
import { loadWords, mergeWords, saveWords } from '../lib/storage.js';

export default function HomeScreen({ onStart, onMyPage }) {
  const fileRef = useRef();
  const [words, setWords] = useState(() => loadWords());
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState('');
  const [quizCount, setQuizCount] = useState(10);
  const [mode, setMode] = useState('mixed');
  const [type, setType] = useState('mixed');

  async function parseFile(file) {
    setError('');
    try {
      const buffer = await file.arrayBuffer();
      const wb = new ExcelJS.Workbook();
      await wb.xlsx.load(buffer);
      const ws = wb.worksheets[0];
      const parsed = [];
      ws.eachRow(row => {
        const en = row.getCell(1).text?.trim();
        const ko = row.getCell(2).text?.trim();
        if (en && ko) parsed.push({ en, ko });
      });
      if (parsed.length === 0) {
        setError('유효한 단어를 찾을 수 없어요. 첫 번째 열: 영어, 두 번째 열: 한국어 형식인지 확인해주세요.');
        return;
      }
      const existing = loadWords();
      const merged = mergeWords(existing, parsed);
      saveWords(merged);
      setWords(merged);
    } catch {
      setError('파일을 읽는 중 오류가 발생했어요.');
    }
  }

  async function onFileChange(e) {
    const file = e.target.files[0];
    if (file) await parseFile(file);
    e.target.value = '';
  }

  async function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) await parseFile(file);
  }

  const canStart = words.length >= 4;

  return (
    <div style={{ maxWidth: 540, margin: '0 auto', padding: '48px 20px 80px', position: 'relative', zIndex: 1 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 40 }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: -0.5, marginBottom: 4 }}>
            💧 WordDrop
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 13 }}>엑셀 업로드 → 단어 퀴즈</p>
        </div>
        <button
          onClick={onMyPage}
          style={{
            background: 'rgba(255,255,255,0.07)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 999,
            color: 'rgba(255,255,255,0.75)',
            padding: '8px 18px',
            fontSize: 13,
            fontWeight: 600,
            display: 'flex', alignItems: 'center', gap: 6,
            transition: 'background 0.2s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.07)'}
        >
          👤 마이페이지
        </button>
      </div>

      {/* Upload zone */}
      <GlassCard
        style={{
          padding: 32,
          border: dragOver
            ? '1.5px dashed rgba(108,143,255,0.7)'
            : '1.5px dashed rgba(255,255,255,0.18)',
          cursor: 'pointer',
          textAlign: 'center',
          transition: 'border-color 0.2s, background 0.2s',
          background: dragOver ? 'rgba(108,143,255,0.08)' : 'rgba(255,255,255,0.04)',
          marginBottom: 20,
        }}
        hover={false}
        onClick={() => fileRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
        <p style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>
          엑셀 파일 업로드
        </p>
        <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 13 }}>
          .xlsx / .xls 파일을 드래그하거나 클릭해서 선택하세요
        </p>
        <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: 12, marginTop: 8 }}>
          1열: 영어 단어 &nbsp;|&nbsp; 2열: 한국어 뜻
        </p>
        <input
          ref={fileRef}
          type="file"
          accept=".xlsx,.xls"
          style={{ display: 'none' }}
          onChange={onFileChange}
        />
      </GlassCard>

      {error && (
        <GlassCard style={{ padding: '12px 16px', marginBottom: 20, border: '1px solid rgba(248,113,113,0.3)', background: 'rgba(248,113,113,0.07)' }} hover={false}>
          <p style={{ color: '#f87171', fontSize: 13 }}>⚠️ {error}</p>
        </GlassCard>
      )}

      {/* Word count pill */}
      {words.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
          <div style={{
            background: 'rgba(74,222,128,0.12)',
            border: '1px solid rgba(74,222,128,0.25)',
            borderRadius: 999, padding: '5px 14px',
            color: '#4ade80', fontSize: 13, fontWeight: 600,
          }}>
            ✓ {words.length}개 단어 로드됨
          </div>
        </div>
      )}

      {/* Quiz settings */}
      {canStart && (
        <GlassCard style={{ padding: 24, marginBottom: 24 }} hover={false}>
          <p style={{ fontWeight: 700, marginBottom: 16, fontSize: 15 }}>퀴즈 설정</p>

          <div style={{ marginBottom: 18 }}>
            <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, fontWeight: 600, letterSpacing: 0.8, textTransform: 'uppercase', marginBottom: 10 }}>
              문항 수
            </p>
            <div style={{ display: 'flex', gap: 8 }}>
              {[5, 10, 20, 'all'].map(n => (
                <button
                  key={n}
                  onClick={() => setQuizCount(n)}
                  style={{
                    padding: '7px 16px', borderRadius: 10, fontSize: 13, fontWeight: 600,
                    background: quizCount === n ? 'rgba(108,143,255,0.25)' : 'rgba(255,255,255,0.06)',
                    border: quizCount === n ? '1px solid rgba(108,143,255,0.5)' : '1px solid rgba(255,255,255,0.1)',
                    color: quizCount === n ? '#6c8fff' : 'rgba(255,255,255,0.6)',
                    transition: 'all 0.15s',
                  }}
                >
                  {n === 'all' ? '전체' : n}
                </button>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: 18 }}>
            <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, fontWeight: 600, letterSpacing: 0.8, textTransform: 'uppercase', marginBottom: 10 }}>
              방향
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {[
                { v: 'mixed', l: '혼합' },
                { v: 'en-to-ko', l: '영→한' },
                { v: 'ko-to-en', l: '한→영' },
              ].map(({ v, l }) => (
                <button
                  key={v}
                  onClick={() => setMode(v)}
                  style={{
                    padding: '7px 16px', borderRadius: 10, fontSize: 13, fontWeight: 600,
                    background: mode === v ? 'rgba(167,139,250,0.2)' : 'rgba(255,255,255,0.06)',
                    border: mode === v ? '1px solid rgba(167,139,250,0.45)' : '1px solid rgba(255,255,255,0.1)',
                    color: mode === v ? '#a78bfa' : 'rgba(255,255,255,0.6)',
                    transition: 'all 0.15s',
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          <div>
            <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, fontWeight: 600, letterSpacing: 0.8, textTransform: 'uppercase', marginBottom: 10 }}>
              퀴즈 형식
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {[
                { v: 'mixed', l: '혼합' },
                { v: 'multiple', l: '객관식' },
                { v: 'typing', l: '주관식' },
              ].map(({ v, l }) => (
                <button
                  key={v}
                  onClick={() => setType(v)}
                  style={{
                    padding: '7px 16px', borderRadius: 10, fontSize: 13, fontWeight: 600,
                    background: type === v ? 'rgba(74,222,128,0.15)' : 'rgba(255,255,255,0.06)',
                    border: type === v ? '1px solid rgba(74,222,128,0.35)' : '1px solid rgba(255,255,255,0.1)',
                    color: type === v ? '#4ade80' : 'rgba(255,255,255,0.6)',
                    transition: 'all 0.15s',
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
        </GlassCard>
      )}

      {/* Start button */}
      {canStart && (
        <button
          onClick={() => onStart(words, quizCount === 'all' ? words.length : quizCount, mode, type)}
          style={{
            width: '100%',
            padding: '16px',
            borderRadius: 16,
            fontSize: 16,
            fontWeight: 700,
            background: 'linear-gradient(135deg, rgba(108,143,255,0.6), rgba(167,139,250,0.5))',
            border: '1px solid rgba(108,143,255,0.4)',
            color: '#fff',
            letterSpacing: 0.3,
            transition: 'opacity 0.2s, transform 0.2s',
          }}
          onMouseEnter={e => { e.currentTarget.style.opacity = '0.85'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          퀴즈 시작 →
        </button>
      )}

      {!canStart && words.length > 0 && (
        <p style={{ textAlign: 'center', color: 'rgba(255,255,255,0.3)', fontSize: 13, marginTop: 12 }}>
          퀴즈를 시작하려면 최소 4개 이상의 단어가 필요해요
        </p>
      )}

      {words.length === 0 && (
        <p style={{ textAlign: 'center', color: 'rgba(255,255,255,0.25)', fontSize: 13, marginTop: 8 }}>
          엑셀 파일을 업로드해서 단어를 불러오세요
        </p>
      )}
    </div>
  );
}
