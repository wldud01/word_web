import { useState } from 'react';
import Blob from './components/Blob.jsx';
import HomeScreen from './screens/HomeScreen.jsx';
import QuizScreen from './screens/QuizScreen.jsx';
import ResultScreen from './screens/ResultScreen.jsx';
import MyPageScreen from './screens/MyPageScreen.jsx';
import { recordResults } from './lib/storage.js';

// screens: 'home' | 'quiz' | 'result' | 'mypage'
export default function App() {
  const [screen, setScreen] = useState('home');
  const [quizParams, setQuizParams] = useState(null);  // { words, count, mode, type }
  const [results, setResults] = useState(null);

  function handleStart(words, count, mode, type) {
    setQuizParams({ words, count, mode, type });
    setScreen('quiz');
  }

  function handleFinish(answers) {
    recordResults(answers.map(a => ({
      en: a.word.en,
      ko: a.word.ko,
      isCorrect: a.isCorrect,
    })));
    setResults(answers);
    setScreen('result');
  }

  function handleRetry() {
    setScreen('quiz');
  }

  return (
    <>
      <Blob />
      <div style={{ position: 'relative', zIndex: 1, minHeight: '100vh' }}>
        {screen === 'home' && (
          <HomeScreen
            onStart={handleStart}
            onMyPage={() => setScreen('mypage')}
          />
        )}
        {screen === 'quiz' && quizParams && (
          <QuizScreen
            {...quizParams}
            onFinish={handleFinish}
            onBack={() => setScreen('home')}
          />
        )}
        {screen === 'result' && results && (
          <ResultScreen
            results={results}
            onRetry={handleRetry}
            onHome={() => setScreen('home')}
            onMyPage={() => setScreen('mypage')}
          />
        )}
        {screen === 'mypage' && (
          <MyPageScreen
            onBack={() => setScreen('home')}
          />
        )}
      </div>
    </>
  );
}
