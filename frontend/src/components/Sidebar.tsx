import {Clapperboard, LayoutGrid, Library, Settings, Sparkles} from 'lucide-react';
import type {View} from '../types';

type Props = {
  view: View;
  onNavigate: (v: View) => void;
  runpodConfigured: boolean;
  running: boolean;
  selectedName: string | null;
};

// 좌측 내비게이션: Projects(생성·관리) / Studio(수정·제작) / Library(에셋) / Settings.
export default function Sidebar({view, onNavigate, runpodConfigured, running, selectedName}: Props) {
  const menu: Array<[View, string, any, string]> = [
    ['projects', 'Projects', LayoutGrid, '생성 · 관리'],
    ['studio', 'Studio', Clapperboard, selectedName ? selectedName : '프로젝트 선택 필요'],
    ['library', 'Library', Library, '로컬 GLB · 에셋 소스'],
  ];
  return (
    <aside>
      <div className="brand">
        <div className="mark"><Sparkles /></div>
        <div><b>AISTUDIO</b><span>AI CHARACTER STUDIO</span></div>
      </div>
      <nav>
        <span className="nav-label">MENU</span>
        {menu.map(([v, label, Icon, sub]) => (
          <button key={v} className={view === v ? 'active' : ''} aria-current={view === v ? 'page' : undefined} onClick={() => onNavigate(v)}>
            <Icon />
            <span className="nav-text"><b>{label}</b><small>{sub}</small></span>
          </button>
        ))}
      </nav>
      <div className="aside-bottom">
        <button className={view === 'settings' ? 'active' : ''} aria-current={view === 'settings' ? 'page' : undefined} onClick={() => onNavigate('settings')}>
          <Settings />Settings
        </button>
        <div className="compute">
          <span><i className={running ? 'busy' : ''} /> {runpodConfigured ? 'RunPod configured' : 'Local compute'}</span>
          <small>GLTF runtime · {running ? 'Busy' : 'Ready'}</small>
        </div>
      </div>
    </aside>
  );
}
