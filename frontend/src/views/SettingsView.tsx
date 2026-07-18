import {useEffect, useState} from 'react';
import {api, errText} from '../api';
import type {RunPodConfig, SystemInfoData} from '../types';

type Props = {
  runpod: RunPodConfig;
  setRunpod: (c: RunPodConfig) => void;
  running: boolean;
  setRunning: (b: boolean) => void;
  notice: string;
  setNotice: (s: string) => void;
};

// RunPod GPU 연결 + 시스템 정보. API key는 Go backend에만 저장된다.
export default function SettingsView({runpod, setRunpod, running, setRunning, notice, setNotice}: Props) {
  const [runpodKey, setRunpodKey] = useState('');
  const [sys, setSys] = useState<SystemInfoData | null>(null);

  useEffect(() => {
    api.systemInfo().then(setSys).catch(() => {});
  }, []);

  async function save() {
    setRunning(true);
    try {
      const status = await api.saveAndTestRunPodConfig(runpod.endpointId, runpodKey, runpod.baseUrl);
      if (status?.ok) {
        const x = await api.getRunPodConfig();
        if (x) setRunpod(x);
        setRunpodKey('');
      }
      setNotice(status?.message || 'RunPod connection verified.');
    } catch (e) {
      setNotice(errText(e));
    } finally {
      setRunning(false);
    }
  }
  async function clear() {
    setRunning(true);
    try {
      const x = await api.clearRunPodConfig();
      if (x) setRunpod(x);
      setRunpodKey('');
      setNotice('저장된 RunPod credential을 삭제했습니다.');
    } catch (e) {
      setNotice(errText(e));
    } finally {
      setRunning(false);
    }
  }
  async function test() {
    setRunning(true);
    try {
      const status = await api.testRunPod();
      setNotice(status?.message || 'RunPod connection verified.');
    } catch (e) {
      setNotice(errText(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="content">
      <div className="settings-card">
        <span className="eyebrow">REMOTE COMPUTE</span>
        <h2>RunPod Serverless</h2>
        <p>API key는 Go backend에만 저장되며 frontend state나 project manifest에 기록되지 않습니다.</p>
        <label>Endpoint ID
          <input value={runpod.endpointId} onChange={e => setRunpod({...runpod, endpointId: e.target.value})} placeholder="예: abcdef123456" />
        </label>
        <label>API key
          <input type="password" autoComplete="new-password" spellCheck={false} value={runpodKey} onChange={e => setRunpodKey(e.target.value)}
            placeholder={runpod.configured ? `Configured via ${runpod.keySource} · 변경할 때만 입력` : 'RunPod Settings에서 생성한 API key 원문'} />
          <small>Endpoint ID, 가려진 **** 값, GitHub/Hugging Face token은 사용할 수 없습니다.</small>
        </label>
        <label>API base URL
          <input value={runpod.baseUrl} onChange={e => setRunpod({...runpod, baseUrl: e.target.value})} />
        </label>
        <div className="settings-actions">
          <button disabled={running || !runpod.configured} onClick={test}>Test connection</button>
          <button disabled={running || !runpod.configured} onClick={clear}>Clear saved key</button>
          <button className="primary" disabled={running || !runpod.endpointId || (!runpodKey && !runpod.configured)} onClick={save}>
            {running ? 'Authenticating…' : 'Verify & save'}
          </button>
        </div>
        {notice && <div className="settings-notice">{notice}</div>}
      </div>
      <div className="settings-card system-card">
        <span className="eyebrow">SYSTEM</span>
        <h2>Runtime environment</h2>
        <div className="sys-rows">
          <div><span>Platform</span><b>{sys?.platform || '—'}</b></div>
          <div><span>Workspace root</span><b>{sys?.workspace || '—'}</b></div>
          <div><span>Projects</span><b>{sys ? sys.jobs : '—'}</b></div>
          <div><span>Python worker</span><b className={sys?.python ? 'ok' : 'warn'}>{sys ? (sys.python ? 'python3 available' : 'python3 missing — local pipeline disabled') : '—'}</b></div>
        </div>
      </div>
    </section>
  );
}
