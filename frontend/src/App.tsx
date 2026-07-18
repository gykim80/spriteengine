import {useEffect, useState} from 'react';
import {Download, FolderOpen, Plus} from 'lucide-react';
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
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [notice, setNotice] = useState('');
  const [running, setRunning] = useState(false);
  const [workerMessage, setWorkerMessage] = useState('');
  const [runpod, setRunpod] = useState<RunPodConfig>({endpointId: '', baseUrl: 'https://api.runpod.ai/v2', configured: false, keySource: 'none'});

  const job = jobs.find(j => j.id === selectedId) || null;

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
  return (
    <div className="shell">
      <Sidebar view={view} onNavigate={setView} runpodConfigured={runpod.configured} running={running} selectedName={job?.name || null} />
      <main>
        <header>
          <div>
            <span className="eyebrow">{eyebrow}</span>
            <h1>{view === 'studio' && job ? job.name : title}</h1>
          </div>
          <div className="header-actions">
            {view === 'studio' && job && (
              <>
                <button onClick={openWorkspace}><FolderOpen />Open workspace</button>
                <button onClick={exportGLB} disabled={running}><Download />Export GLB</button>
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
            onRename={renameJob} onDelete={deleteJob} setNotice={setNotice} />
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
