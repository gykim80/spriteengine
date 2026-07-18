// 텍스트 프롬프트 → 모션 연출 스펙 파서.
// three.js에 의존하지 않아 메인 번들에 안전하게 포함된다 (clip 합성은 motionClip.ts).

export type MotionActionId =
  | 'fly' | 'kick' | 'punch' | 'jump' | 'spin' | 'dash'
  | 'run' | 'walk' | 'wave' | 'bow' | 'dance' | 'idle';

export type MotionAction = {id: MotionActionId; label: string; repeat: number};

export type MotionSpec = {
  prompt: string;
  actions: MotionAction[];
  /** 전체 재생 속도 배율. "빠르게"=1.5, "천천히"=0.65, 기본 1 */
  tempo: number;
  /** tempo가 1이 아닐 때 UI에 표시할 수식어 라벨 */
  tempoLabel?: string;
  /** viewport clip 목록에 표시되고 선택에 쓰이는 이름 */
  clipName: string;
};

// 한국어/영어 키워드 사전. 문장 내 등장 순서대로 동작을 시퀀싱한다.
const LEXICON: {id: MotionActionId; label: string; pattern: RegExp}[] = [
  {id: 'fly', label: '비행', pattern: /날아|날으|날며|비행|활공|fly|flying|soar/i},
  {id: 'kick', label: '발차기', pattern: /발차기|발길질|차기|차는|킥|kick/i},
  {id: 'punch', label: '펀치', pattern: /펀치|주먹|때리|타격|punch|hit/i},
  {id: 'jump', label: '점프', pattern: /점프|뛰어|도약|jump|hop|leap/i},
  {id: 'spin', label: '회전', pattern: /회전|돌아|돌면|스핀|빙글|spin|rotate|turn/i},
  {id: 'dash', label: '돌진', pattern: /돌진|대시|질주|급습|dash|charge|rush|lunge/i},
  {id: 'run', label: '달리기', pattern: /달리|달려|런닝|run|sprint|jog/i},
  {id: 'walk', label: '걷기', pattern: /걷|걸어|산책|walk|stroll/i},
  {id: 'wave', label: '인사', pattern: /손\s*흔들|인사|안녕|반갑|wave|hello|greet/i},
  {id: 'bow', label: '절', pattern: /절하|절을|꾸벅|목례|bow/i},
  {id: 'dance', label: '춤', pattern: /춤|댄스|dance|groove/i},
  {id: 'idle', label: '대기', pattern: /대기|숨쉬|가만히|idle|breath|stand/i},
];

const MAX_ACTIONS = 4;
const MAX_REPEAT = 5;

// 반복 횟수 수식어: "두 번", "3번", "x2", "twice", "three times" 등
const COUNT_PATTERN = /(\d+)\s*(?:번|회|차례|\s?times?)|(한|두|세|네|다섯)\s*(?:번|회|차례)|\b(once|twice|thrice)\b|[x×]\s*(\d+)/gi;
const KOREAN_NUMERALS: Record<string, number> = {한: 1, 두: 2, 세: 3, 네: 4, 다섯: 5};
const ENGLISH_COUNTS: Record<string, number> = {once: 1, twice: 2, thrice: 3};

// 전체 속도 수식어 (마지막 언급이 우선)
const TEMPO_FAST = /빨리|빠르게|빠른|신속|재빨리|재빠르|급하게|quick|fast|swift|rapid/i;
const TEMPO_SLOW = /천천히|느리게|느린|느릿|슬로우|slowly|slow|gentle/i;

/** 프롬프트에서 동작을 추출한다. 인식된 동작이 없으면 null. */
export function parseMotionPrompt(text: string): MotionSpec | null {
  const prompt = text.trim();
  if (!prompt) return null;
  const found: {index: number; action: MotionAction}[] = [];
  for (const entry of LEXICON) {
    const index = prompt.search(entry.pattern);
    if (index >= 0) found.push({index, action: {id: entry.id, label: entry.label, repeat: 1}});
  }
  if (!found.length) return null;
  // 문장 내 등장 순서 → 연출 순서 ("날아서 발차기" = fly → kick)
  found.sort((a, b) => a.index - b.index);
  const seen = new Set<MotionActionId>();
  const kept = found
    .filter(f => !seen.has(f.action.id) && (seen.add(f.action.id), true))
    .slice(0, MAX_ACTIONS);

  // 반복 수식어를 가장 가까운 동작 키워드에 귀속시킨다 ("두 번 발차기하고 세 번 점프")
  for (const m of prompt.matchAll(COUNT_PATTERN)) {
    const value = m[1] ? parseInt(m[1], 10)
      : m[2] ? KOREAN_NUMERALS[m[2]]
      : m[3] ? ENGLISH_COUNTS[m[3].toLowerCase()]
      : parseInt(m[4], 10);
    if (!value || value < 1) continue;
    let best = kept[0];
    let bestDist = Infinity;
    for (const f of kept) {
      const dist = Math.abs(f.index - (m.index ?? 0));
      if (dist < bestDist) { bestDist = dist; best = f; }
    }
    best.action.repeat = Math.min(value, MAX_REPEAT);
  }

  // 속도 수식어: 빠르게/천천히 모두 있으면 뒤에 나온 쪽이 이긴다
  const fastIdx = prompt.search(TEMPO_FAST);
  const slowIdx = prompt.search(TEMPO_SLOW);
  let tempo = 1;
  let tempoLabel: string | undefined;
  if (fastIdx >= 0 && (slowIdx < 0 || fastIdx > slowIdx)) { tempo = 1.5; tempoLabel = '빠르게'; }
  else if (slowIdx >= 0) { tempo = 0.65; tempoLabel = '천천히'; }

  const actions = kept.map(f => f.action);
  const seq = actions.map(a => (a.repeat > 1 ? `${a.label}×${a.repeat}` : a.label)).join(' → ');
  return {
    prompt,
    actions,
    tempo,
    tempoLabel,
    clipName: `연출 · ${seq}${tempoLabel ? ` · ${tempoLabel}` : ''}`,
  };
}
