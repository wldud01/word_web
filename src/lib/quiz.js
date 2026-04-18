// Shuffle array (Fisher-Yates)
export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Build quiz questions from word list
// mode: 'en-to-ko' | 'ko-to-en' | 'mixed'
// type: 'multiple' | 'typing' | 'mixed'
export function buildQuiz(words, count = 10, mode = 'mixed', type = 'mixed') {
  if (words.length < 4) return [];

  // Prioritise words with high wrong rate or never seen
  const sorted = [...words].sort((a, b) => {
    const scoreA = a.lastSeen ? a.wrong / (a.correct + a.wrong + 1) : 1;
    const scoreB = b.lastSeen ? b.wrong / (b.correct + b.wrong + 1) : 1;
    return scoreB - scoreA;
  });

  const pool = shuffle(sorted).slice(0, Math.min(count, words.length));

  return pool.map((word, i) => {
    const qMode = mode === 'mixed'
      ? (i % 2 === 0 ? 'en-to-ko' : 'ko-to-en')
      : mode;

    const qType = type === 'mixed'
      ? (i % 3 === 0 ? 'typing' : 'multiple')
      : type;

    const question = qMode === 'en-to-ko' ? word.en : word.ko;
    const answer = qMode === 'en-to-ko' ? word.ko : word.en;
    const label = qMode === 'en-to-ko' ? '한국어 뜻을 고르세요' : '영어 단어를 고르세요';

    let choices = null;
    if (qType === 'multiple') {
      const wrong = shuffle(words.filter(w => w.en !== word.en))
        .slice(0, 3)
        .map(w => qMode === 'en-to-ko' ? w.ko : w.en);
      choices = shuffle([answer, ...wrong]);
    }

    return { id: i, word, question, answer, label, type: qType, choices };
  });
}
