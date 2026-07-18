// 텍스트 프롬프트 → 모션 연출 스펙 파서.
// three.js에 의존하지 않아 메인 번들에 안전하게 포함된다 (clip 합성은 motionClip.ts).

export type MotionActionId =
  | 'fly' | 'kick' | 'punch' | 'jump' | 'spin' | 'dash'
  | 'run' | 'walk' | 'wave' | 'bow' | 'dance' | 'idle';

export type MotionAction = {id: MotionActionId; label: string};

export type MotionSpec = {
  prompt: string;
  actions: MotionAction[];
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

/** 프롬프트에서 동작을 추출한다. 인식된 동작이 없으면 null. */
export function parseMotionPrompt(text: string): MotionSpec | null {
  const prompt = text.trim();
  if (!prompt) return null;
  const found: {index: number; action: MotionAction}[] = [];
  for (const entry of LEXICON) {
    const index = prompt.search(entry.pattern);
    if (index >= 0) found.push({index, action: {id: entry.id, label: entry.label}});
  }
  if (!found.length) return null;
  // 문장 내 등장 순서 → 연출 순서 ("날아서 발차기" = fly → kick)
  found.sort((a, b) => a.index - b.index);
  const seen = new Set<MotionActionId>();
  const actions = found
    .filter(f => !seen.has(f.action.id) && (seen.add(f.action.id), true))
    .slice(0, MAX_ACTIONS)
    .map(f => f.action);
  return {
    prompt,
    actions,
    clipName: `연출 · ${actions.map(a => a.label).join(' → ')}`,
  };
}
