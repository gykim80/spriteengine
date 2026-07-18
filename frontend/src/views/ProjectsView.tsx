import {useEffect, useRef, useState} from 'react';
import {Check, FolderOpen, ImagePlus, Layers3, MoreHorizontal, Pencil, Plus, Trash2, X} from 'lucide-react';
import {api} from '../api';
import type {Job} from '../types';

type Props = {
  jobs: Job[];
  running: boolean;
  onOpen: (id: string) => void;
  onImport: () => void;
  onImportFile: (file: File) => Promise<void>;
  onRename: (id: string, name: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  setNotice: (s: string) => void;
};

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

const statusLabel: Record<string, string> = {
  draft: 'Draft', ready: 'Ready', processing: 'Processing', complete: 'Complete', failed: 'Failed',
};

// 홈 화면: 프로젝트 생성 + 카드 그리드 관리(열기/이름변경/삭제/워크스페이스).
export default function ProjectsView({jobs, running, onOpen, onImport, onImportFile, onRename, onDelete, setNotice}: Props) {
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const renameRef = useRef<HTMLInputElement>(null);

  // 이미지 파일 drag & drop → 즉시 프로젝트 생성
  async function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (running) return;
    const file = Array.from(e.dataTransfer.files).find(f => IMAGE_EXT.test(f.name));
    if (!file) {
      setNotice('PNG, JPG, WEBP 이미지 파일만 드롭할 수 있습니다.');
      return;
    }
    await onImportFile(file);
  }

  // 참조 이미지 썸네일을 backend에서 data URI로 로드 (실패 시 gradient fallback 유지)
  useEffect(() => {
    let alive = true;
    jobs.forEach(j => {
      if (!j.image || thumbs[j.id]) return;
      api.readJobImage(j.id)
        .then(uri => { if (alive && uri) setThumbs(t => ({...t, [j.id]: uri})); })
        .catch(() => {});
    });
    return () => { alive = false; };
  }, [jobs]);

  useEffect(() => { if (renaming) renameRef.current?.select(); }, [renaming]);

  function beginRename(j: Job) {
    setRenaming(j.id);
    setDraft(j.name);
    setMenuFor(null);
    setConfirmDelete(null);
  }
  async function commitRename(id: string) {
    const name = draft.trim();
    setRenaming(null);
    if (!name) return;
    await onRename(id, name);
  }
  async function openWorkspace(id: string) {
    setMenuFor(null);
    try { await api.openWorkspace(id); } catch (e) { setNotice(String(e)); }
  }

  return (
    <section className="content"
      onDragOver={e => { e.preventDefault(); if (!running) setDragOver(true); }}
      onDragLeave={e => { if (e.currentTarget === e.target) setDragOver(false); }}
      onDrop={handleDrop}>
      <div className="projects-head">
        <div>
          <span className="eyebrow">ALL PROJECTS</span>
          <b>{jobs.length ? `${jobs.length} character project${jobs.length > 1 ? 's' : ''}` : 'No projects yet'}</b>
        </div>
        <small>이미지 한 장으로 프로젝트를 생성하고, 카드에서 관리하고, Studio에서 편집합니다.</small>
      </div>
      <div className="project-grid">
        <button className={`project-new ${dragOver ? 'drag-over' : ''}`} onClick={onImport} disabled={running}>
          <div className="new-icon"><ImagePlus /><Plus /></div>
          <b>{dragOver ? '여기에 놓아 프로젝트 생성' : 'Import reference image'}</b>
          <small>{dragOver ? '이미지에서 즉시 파이프라인 준비' : 'PNG, JPG, WEBP · 클릭 또는 drag & drop'}</small>
        </button>
        {jobs.map(j => {
          const running_ = j.status === 'processing';
          return (
            <div key={j.id} className={`project-card ${running_ ? 'busy' : ''}`}>
              <button className="project-open" onClick={() => onOpen(j.id)} aria-label={`Open ${j.name} in Studio`}>
                <div className="project-thumb">
                  {thumbs[j.id] ? <img src={thumbs[j.id]} alt="" /> : <Layers3 />}
                  <span className={`badge ${j.status}`}>{statusLabel[j.status] || j.status}</span>
                </div>
                <div className="project-progress"><i style={{width: `${j.progress}%`}} /></div>
              </button>
              <div className="project-meta">
                {renaming === j.id ? (
                  <div className="inline-rename">
                    <input ref={renameRef} value={draft} onChange={e => setDraft(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') commitRename(j.id); if (e.key === 'Escape') setRenaming(null); }} />
                    <button aria-label="Confirm rename" onClick={() => commitRename(j.id)}><Check /></button>
                    <button aria-label="Cancel rename" onClick={() => setRenaming(null)}><X /></button>
                  </div>
                ) : (
                  <>
                    <div className="project-title">
                      <b title={j.name}>{j.name}</b>
                      <small>{j.progress}% · {j.artifacts?.length || 0} artifacts · {new Date(j.created).toLocaleDateString()}</small>
                    </div>
                    <button className="card-menu-btn" aria-label="Project options"
                      onClick={() => { setMenuFor(menuFor === j.id ? null : j.id); setConfirmDelete(null); }}>
                      <MoreHorizontal />
                    </button>
                  </>
                )}
                {menuFor === j.id && (
                  <div className="card-menu" role="menu">
                    <button onClick={() => beginRename(j)}><Pencil />Rename</button>
                    <button onClick={() => openWorkspace(j.id)}><FolderOpen />Open workspace</button>
                    {confirmDelete === j.id ? (
                      <button className="danger" disabled={running_}
                        onClick={async () => { setMenuFor(null); setConfirmDelete(null); await onDelete(j.id); }}>
                        <Trash2 />정말 삭제할까요?
                      </button>
                    ) : (
                      <button className="danger" disabled={running_} onClick={() => setConfirmDelete(j.id)}>
                        <Trash2 />{running_ ? 'Processing 중에는 불가' : 'Delete project'}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
