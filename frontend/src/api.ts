import type {Job, RunPodConfig, RunPodStatus, SystemInfoData} from './types';

// Wails가 window.go.main.App에 바인딩한 Go 메서드에 대한 타입 안전 래퍼.
const app = () => (window as any).go?.main?.App;

async function call<T>(fn: (a: any) => Promise<T>): Promise<T> {
  const a = app();
  if (!a) throw new Error('backend is not ready');
  return fn(a);
}

export const api = {
  listJobs: () => call<Job[]>(a => a.ListJobs()),
  importReference: () => call<Job>(a => a.ImportReference()),
  importReferenceData: (filename: string, base64: string) => call<Job>(a => a.ImportReferenceData(filename, base64)),
  runNextStage: (id: string) => call<Job>(a => a.RunNextStage(id)),
  runAllStages: (id: string) => call<Job>(a => a.RunAllStages(id)),
  deleteJob: (id: string) => call<Job[]>(a => a.DeleteJob(id)),
  renameJob: (id: string, name: string) => call<Job>(a => a.RenameJob(id, name)),
  resetStage: (id: string, stageId: string) => call<Job>(a => a.ResetStage(id, stageId)),
  exportFinalGLB: (id: string) => call<string>(a => a.ExportFinalGLB(id)),
  readJobImage: (id: string) => call<string>(a => a.ReadJobImage(id)),
  readArtifact: (path: string) => call<string>(a => a.ReadArtifact(path)),
  openWorkspace: (id: string) => call<void>(a => a.OpenWorkspace(id)),
  openExternal: (url: string) => call<void>(a => a.OpenExternal(url)),
  getRunPodConfig: () => call<RunPodConfig>(a => a.GetRunPodConfig()),
  saveAndTestRunPodConfig: (endpointId: string, apiKey: string, baseUrl: string) =>
    call<RunPodStatus>(a => a.SaveAndTestRunPodConfig(endpointId, apiKey, baseUrl)),
  testRunPod: () => call<RunPodStatus>(a => a.TestRunPod()),
  clearRunPodConfig: () => call<RunPodConfig>(a => a.ClearRunPodConfig()),
  systemInfo: () => call<SystemInfoData>(a => a.SystemInfo()),
};

export const isCancelled = (e: unknown) => String(e).toLowerCase().includes('cancelled');
export const errText = (e: unknown) => String(e).replace(/^Error:\s*/, '');
