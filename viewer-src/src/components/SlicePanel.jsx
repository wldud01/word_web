import React, { useEffect, useRef } from 'react';
import { renderThumbnail } from '../lib/ctRenderer';

export default function SlicePanel({ slices, currentIndex, onSelect }) {
  return (
    <div style={{
      width: 110,
      height: '100%',
      background: '#0d0d1a',
      borderRight: '1px solid #2a2a4a',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      paddingTop: 8,
      gap: 4,
    }}>
      {slices.map((slice, i) => (
        <SliceThumbnail
          key={i}
          slice={slice}
          index={i}
          isActive={i === currentIndex}
          onClick={() => onSelect(i)}
        />
      ))}
    </div>
  );
}

function SliceThumbnail({ slice, index, isActive, onClick }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (canvasRef.current && slice) {
      renderThumbnail(canvasRef.current, slice.data, slice.width, slice.height);
    }
  }, [slice]);

  return (
    <div
      onClick={onClick}
      style={{
        cursor: 'pointer',
        border: isActive ? '2px solid #4fc3f7' : '2px solid transparent',
        borderRadius: 4,
        padding: 2,
        background: isActive ? '#1a2a3a' : 'transparent',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        flexShrink: 0,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: 80, height: 80, display: 'block', borderRadius: 2 }}
      />
      <span style={{ fontSize: 10, color: isActive ? '#4fc3f7' : '#888', fontFamily: 'monospace' }}>
        #{String(index + 1).padStart(3, '0')}
      </span>
    </div>
  );
}
