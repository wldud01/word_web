import React, { useMemo } from 'react';
import { computePixelStats } from '../lib/mockData';
import VolumeRenderer3D from './VolumeRenderer3D';

export default function InfoPanel({ patient, sliceIndex, sliceData, slices }) {
  const stats = useMemo(() => {
    if (!sliceData) return null;
    return computePixelStats(sliceData.data);
  }, [sliceData]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#0f0f1f', borderRight: '1px solid #2a2a4a', width: 220 }}>

      {/* Patient Info */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a2a4a' }}>
        <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', marginBottom: 4 }}>환자 정보</div>
        <div style={row}><span style={lbl}>ID</span><span style={{ ...val, fontSize: 10 }}>{patient.id}</span></div>
        <div style={row}><span style={lbl}>Modality</span><span style={val}>{patient.modality}</span></div>
        <div style={row}><span style={lbl}>총 슬라이스</span><span style={val}>{patient.sliceCount}</span></div>
        <div style={row}><span style={lbl}>두께</span><span style={val}>{patient.sliceThickness} mm</span></div>
      </div>

      {/* Slice stats */}
      <div style={{ padding: '10px 12px', flexShrink: 0 }}>
        <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', marginBottom: 8 }}>슬라이스 정보</div>
        <div style={row}><span style={lbl}>슬라이스 #</span><span style={val}>{sliceIndex + 1} / {patient.sliceCount}</span></div>
        <div style={row}><span style={lbl}>위치</span><span style={val}>{(sliceIndex * patient.sliceThickness).toFixed(1)} mm</span></div>

        {stats && (
          <>
            <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', margin: '10px 0 6px' }}>픽셀 통계</div>
            <div style={row}><span style={lbl}>최솟값</span><span style={{ ...val, color: '#81d4fa' }}>{stats.min}</span></div>
            <div style={row}><span style={lbl}>최댓값</span><span style={{ ...val, color: '#f48fb1' }}>{stats.max}</span></div>
            <div style={row}><span style={lbl}>평균</span><span style={{ ...val, color: '#a5d6a7' }}>{stats.mean}</span></div>
            <div style={row}><span style={lbl}>표준편차</span><span style={{ ...val, color: '#ffcc80' }}>{stats.std}</span></div>

            <div style={{ marginTop: 12 }}>
              <div style={{ color: '#666', fontSize: 10, marginBottom: 4 }}>MRI T1 참조값</div>
              {[
                { label: '배경',   range: '0',         color: '#444' },
                { label: 'CSF',    range: '20 ~ 80',   color: '#64b5f6' },
                { label: '회백질', range: '80 ~ 140',  color: '#81c784' },
                { label: '백질',   range: '140 ~ 215', color: '#e0e0e0' },
                { label: '고강도', range: '215 ~ 255', color: '#ffcc80' },
              ].map(({ label, range, color }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid #1a1a2e' }}>
                  <span style={{ fontSize: 10, color }}>{label}</span>
                  <span style={{ fontSize: 10, color: '#888', fontFamily: 'monospace' }}>{range}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* 남는 공간에 3D 렌더링을 작게 끼워넣음 */}
      <div style={{ flex: 1, minHeight: 140, borderTop: '1px solid #2a2a4a', overflow: 'hidden' }}>
        <VolumeRenderer3D slices={slices} sliceIndex={sliceIndex} />
      </div>
    </div>
  );
}

const row = { display: 'flex', justifyContent: 'space-between', padding: '2px 0', alignItems: 'center' };
const lbl = { fontSize: 11, color: '#888' };
const val = { fontSize: 11, color: '#e0e0e0', fontFamily: 'monospace' };
