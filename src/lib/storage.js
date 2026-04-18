const KEY_HISTORY = 'vq_history';
const KEY_WORDS = 'vq_words';

// Word record: { en, ko, correct: number, wrong: number, lastSeen: timestamp }

export function loadWords() {
  try {
    return JSON.parse(localStorage.getItem(KEY_WORDS) || '[]');
  } catch {
    return [];
  }
}

export function saveWords(words) {
  localStorage.setItem(KEY_WORDS, JSON.stringify(words));
}

// Merge newly uploaded words with existing records
export function mergeWords(existing, incoming) {
  const map = new Map(existing.map(w => [w.en.toLowerCase(), w]));
  for (const { en, ko } of incoming) {
    const key = en.toLowerCase();
    if (!map.has(key)) {
      map.set(key, { en, ko, correct: 0, wrong: 0, lastSeen: null });
    } else {
      // Update Korean in case it changed
      map.get(key).ko = ko;
    }
  }
  return Array.from(map.values());
}

export function recordResults(results) {
  // results: [{ en, ko, isCorrect }]
  const words = loadWords();
  const map = new Map(words.map(w => [w.en.toLowerCase(), w]));
  for (const { en, ko, isCorrect } of results) {
    const key = en.toLowerCase();
    if (!map.has(key)) {
      map.set(key, { en, ko, correct: 0, wrong: 0, lastSeen: null });
    }
    const w = map.get(key);
    if (isCorrect) w.correct += 1;
    else w.wrong += 1;
    w.lastSeen = Date.now();
  }
  const updated = Array.from(map.values());
  saveWords(updated);

  // Also append to history sessions
  const history = loadHistory();
  history.unshift({
    id: Date.now(),
    date: new Date().toISOString(),
    total: results.length,
    correct: results.filter(r => r.isCorrect).length,
    results,
  });
  localStorage.setItem(KEY_HISTORY, JSON.stringify(history.slice(0, 50)));
}

export function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY_HISTORY) || '[]');
  } catch {
    return [];
  }
}

export function clearAll() {
  localStorage.removeItem(KEY_HISTORY);
  localStorage.removeItem(KEY_WORDS);
}
