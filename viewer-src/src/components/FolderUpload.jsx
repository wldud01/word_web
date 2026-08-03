import React, { useRef } from 'react';
import { filesFromFileList } from '../lib/patientFolder';

// 헤더에 들어가는 작은 "폴더 업로드" 버튼. 클릭하면 OS 폴더 선택창이 뜬다.
export default function FolderUpload({ onFilesSelected, disabled }) {
  const inputRef = useRef(null);

  const onChange = (e) => {
    const files = filesFromFileList(e.target.files);
    e.target.value = '';
    if (files.length > 0) onFilesSelected(files);
  };

  return (
    <>
      <button
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        style={{
          padding: '5px 14px', fontSize: 12, fontWeight: 'bold',
          background: disabled ? '#1a1a2a' : '#1a2a3a',
          color: disabled ? '#555' : '#64b5f6',
          border: `1px solid ${disabled ? '#2a2a3a' : '#3a5a7a'}`,
          borderRadius: 4, cursor: disabled ? 'default' : 'pointer',
        }}
      >
        📂 환자 폴더 업로드
      </button>
      <input
        ref={inputRef}
        type="file"
        // @ts-ignore
        webkitdirectory=""
        directory=""
        multiple
        style={{ display: 'none' }}
        onChange={onChange}
      />
    </>
  );
}
