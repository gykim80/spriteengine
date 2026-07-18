import {useEffect, useState} from 'react';
import {Check, Download, FolderOpen, Pencil, Plus, X} from 'lucide-react';
import {EventsOn} from '../wailsjs/runtime/runtime';
import {api, errText, isCancelled} from './api';
import type {Job, RunPodConfig, View} from './types';
import Sidebar from './components/Sidebar';
import ProjectsView from './views/ProjectsView';
import StudioView from './views/StudioView';
import LibraryView from './views/LibraryView';
import SettingsView from './views/SettingsView';

const headerCopy: Record<View, [string, string]> = {
  projects: ['WORKSPACE / PROJECTS', 'Turn one image into a living character.'],
  studio: ['WORKSPACE / STUDIO', 'Studio'],
  library: ['WORKSPACE / LIBRARY', 'Preview real skeletal clips and asset sources.'],
  settings: ['WORKSPACE / SETTINGS', 'RunPod GPU connection'],
};

function App() {
  const [view, setView] = useState<View>('projects');
  // 마지막으로 열었던 프로젝트를 재시작 후에도 복원 (없는 id는 job 조회에서 자연히 무시됨)
  const [selectedId, setSelectedId] = useState<string | null>(() => localStorage.getItem('se:lastProject'));
  const [jobs, setJobs] = useState<Job[]>([]);
  const [notice, setNotice] = useState('');
  const [running, setRunning] = useState(false);
  const [workerMessage, setWorkerMessage] = useState('');
  const [runpod, setRunpod] = useState<RunPodConfig>({endpointId: '', baseUrl: 'https://api.runpod.ai/v2', configured: false, keySource: 'none'});
  const [renamingHead, setRenamingHead] = useState(false);
  const [headDraft, setHeadDraft] = useState('');

  const job = jobs.find(j => j.id === selectedId) || null;

  // view나 프로젝트가 바뀌면 헤더 rename 편집 상태를 닫는다.
  useEffect(() => { setRenamingHead(false); }, [view, selectedId]);

  useEffect(() => {
    if (selectedId) localStorage.setItem('se:lastProject', selectedId);
    else localStorage.removeItem('se:lastProject');
  }, [selectedId]);

  useEffect(() => {
    api.listJobs().then(x => setJobs(x || [])).catch(() => {});
    api.getRunPodConfig().then(x => x && setRunpod(x)).catch(() => {});
    let off = () => {};
    let offJob = () => {};
    try {
      off = EventsOn('worker:event', (event: any) => { if (event?.message) setWorkerMessage(event.message); });
      // 실행 중인 pipeline의 stage 상태를 실시간 반영 (RunAllStages 완료 전에도 UI 갱신)
      offJob = EventsOn('job:update', (j: Job) => { if (j?.id) setJobs(v => v.map(x => (x.id === j.id ? j : x))); });
    } catch {}
    return () => { off(); offJob(); };
  }, []);

  // notice는 8초 뒤 자동으로 사라진다 (클릭으로 즉시 닫기도 유지).
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(''), 8000);
    return () => clearTimeout(t);
  }, [notice]);

  const updateJob = (j: Job) => setJobs(v => v.map(x => (x.id === j.id ? j : x)));

  async function importImage() {
    try {
      const j = await api.importReference();
      if (j) {
        setJobs(v => [j, ...v.filter(x => x.id !== j.id)]);
        setSelectedId(j.id);
        setView('studio');
        setNotice('Reference copied with SHA-256 provenance. Pipeline을 실행하세요.');
      }
    } catch (e) {
      if (!isCancelled(e)) setNotice(errText(e));
    }
  }
  // drag & drop된 이미지 File을 base64로 backend에 전달해 프로젝트 생성
  async function importImageFile(file: File) {
    try {
      const b64 = await new Promise<string>((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result).split(',')[1] || '');
        r.onerror = () => reject(new Error('파일을 읽지 못했습니다'));
        r.readAsDataURL(file);
      });
      const j = await api.importReferenceData(file.name, b64);
      if (j) {
        setJobs(v => [j, ...v.filter(x => x.id !== j.id)]);
        setSelectedId(j.id);
        setView('studio');
        setNotice('Reference copied with SHA-256 provenance. Pipeline을 실행하세요.');
      }
    } catch (e) {
      setNotice(errText(e));
    }
  }
  async function runNext() {
    if (!job) return;
    setRunning(true);
    setWorkerMessage('Starting isolated worker…');
    try {
      const j = await api.runNextStage(job.id);
      if (j) updateJob(j);
      setNotice('Stage completed and artifact provenance was recorded.');
    } catch (e) {
      setNotice(errText(e));
      api.listJobs().then(x => setJobs(x || [])).catch(() => {});
    } finally {
      setRunning(false);
      setWorkerMessage('');
    }
  }
  async function runAll() {
    if (!job) return;
    setRunning(true);
    setWorkerMessage('Running complete pipeline…');
    try {
      const j = await api.runAllStages(job.id);
      if (j) updateJob(j);
      setNotice('Complete animation-ready GLB pipeline finished.');
    } catch (e) {
      setNotice(errText(e));
      api.listJobs().then(x => setJobs(x || [])).catch(() => {});
    } finally {
      setRunning(false);
      setWorkerMessage('');
    }
  }
  async function resetStage(stageId: string) {
    if (!job) return;
    try {
      const j = await api.resetStage(job.id, stageId);
      if (j) updateJob(j);
      setNotice(`${stageId} 단계부터 다시 실행할 수 있습니다. 하위 단계 아티팩트는 정리되었습니다.`);
    } catch (e) {
      setNotice(errText(e));
    }
  }
  async function deleteJob(id: string) {
    try {
      const list = await api.deleteJob(id);
      setJobs(list || []);
      if (selectedId === id) setSelectedId(null);
      setNotice('프로젝트와 워크스페이스 파일을 삭제했습니다.');
    } catch (e) {
      setNotice(errText(e));
    }
  }
  async function renameJob(id: string, name: string) {
    try {
      const j = await api.renameJob(id, name);
      if (j) updateJob(j);
    } catch (e) {
      setNotice(errText(e));
    }
  }
  // Studio 헤더 제목에서 바로 이름 변경 (Projects 카드 메뉴와 동일 동작)
  function beginHeadRename() {
    if (!job) return;
    setHeadDraft(job.name);
    setRenamingHead(true);
  }
  async function commitHeadRename() {
    const name = headDraft.trim();
    setRenamingHead(false);
    if (!job || !name || name === job.name) return;
    await renameJob(job.id, name);
  }
  async function exportGLB() {
    if (!job) return;
    try {
      const dst = await api.exportFinalGLB(job.id);
      if (dst) setNotice(`GLB 저장 완료 · ${dst}`);
    } catch (e) {
      if (!isCancelled(e)) setNotice(errText(e));
    }
  }
  async function openWorkspace() {
    if (!job) return;
    try { await api.openWorkspace(job.id); } catch (e) { setNotice(errText(e)); }
  }
  function openProject(id: string) {
    setSelectedId(id);
    setView('studio');
  }

  const [eyebrow, title] = headerCopy[view];
  const hasGLB = !!job?.artifacts?.some(a => a.path.toLowerCase().endsWith('.glb'));
  return (
    <div className="shell">
      <Sidebar view={view} onNavigate={setView} runpodConfigured={runpod.configured} running={running} selectedName={job?.name || null} />
      <main>
        <header>
          <div>
            <span className="eyebrow">{eyebrow}</span>
            {view === 'studio' && job ? (
              renamingHead ? (
                <div className="inline-rename head-rename">
                  <input autoFocus value={headDraft} onChange={e => setHeadDraft(e.target.value)} aria-label="Project name"
                    onKeyDown={e => { if (e.key === 'Enter') commitHeadRename(); if (e.key === 'Escape') setRenamingHead(false); }} />
                  <button aria-label="Confirm rename" onClick={commitHeadRename}><Check /></button>
                  <button aria-label="Cancel rename" onClick={() => setRenamingHead(false)}><X /></button>
                </div>
              ) : (
                <h1 className="head-title">
                  {job.name}
                  <button className="head-rename-btn" aria-label="Rename project" title="Rename project" onClick={beginHeadRename}><Pencil /></button>
                </h1>
              )
            ) : (
              <h1>{title}</h1>
            )}
          </div>
          <div className="header-actions">
            {view === 'studio' && job && (
              <>
                <button onClick={openWorkspace}><FolderOpen />Open workspace</button>
                <button onClick={exportGLB} disabled={running || !hasGLB}
                  title={hasGLB ? undefined : 'GLB artifact 없음 · pipeline 먼저 실행'}><Download />Export GLB</button>
              </>
            )}
            {view !== 'settings' && (
              <button className="primary" onClick={importImage} disabled={running}><Plus />Import character</button>
            )}
          </div>
        </header>
        {notice && view !== 'settings' && (
          <div className="notice-wrap">
            <button className="notice" onClick={() => setNotice('')}>{notice}<span>×</span></button>
          </div>
        )}
        {view === 'projects' && (
          <ProjectsView jobs={jobs} running={running} onOpen={openProject} onImport={importImage}
            onImportFile={importImageFile} onRename={renameJob} onDelete={deleteJob} setNotice={setNotice} />
        )}
        {view === 'studio' && (
          <StudioView job={job} running={running} workerMessage={workerMessage}
            onRunNext={runNext} onRunAll={runAll} onReset={resetStage} onExport={exportGLB}
            goProjects={() => setView('projects')} setNotice={setNotice} />
        )}
        {view === 'library' && <LibraryView setNotice={setNotice} />}
        {view === 'settings' && (
          <SettingsView runpod={runpod} setRunpod={setRunpod} running={running} setRunning={setRunning}
            notice={notice} setNotice={setNotice} />
        )}
      </main>
    </div>
  );
}

export default App;
