export namespace main {
	
	export class Artifact {
	    stage: string;
	    kind: string;
	    path: string;
	    metrics?: Record<string, any>;
	
	    static createFrom(source: any = {}) {
	        return new Artifact(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.stage = source["stage"];
	        this.kind = source["kind"];
	        this.path = source["path"];
	        this.metrics = source["metrics"];
	    }
	}
	export class LogEntry {
	    time: string;
	    stage: string;
	    level: string;
	    message: string;
	
	    static createFrom(source: any = {}) {
	        return new LogEntry(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.time = source["time"];
	        this.stage = source["stage"];
	        this.level = source["level"];
	        this.message = source["message"];
	    }
	}
	export class Stage {
	    id: string;
	    name: string;
	    status: string;
	    detail: string;
	
	    static createFrom(source: any = {}) {
	        return new Stage(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.status = source["status"];
	        this.detail = source["detail"];
	    }
	}
	export class Job {
	    id: string;
	    name: string;
	    created: string;
	    status: string;
	    progress: number;
	    image?: string;
	    imageHash?: string;
	    workspace?: string;
	    stages: Stage[];
	    artifacts?: Artifact[];
	    logs?: LogEntry[];
	
	    static createFrom(source: any = {}) {
	        return new Job(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.created = source["created"];
	        this.status = source["status"];
	        this.progress = source["progress"];
	        this.image = source["image"];
	        this.imageHash = source["imageHash"];
	        this.workspace = source["workspace"];
	        this.stages = this.convertValues(source["stages"], Stage);
	        this.artifacts = this.convertValues(source["artifacts"], Artifact);
	        this.logs = this.convertValues(source["logs"], LogEntry);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	export class RunPodConfig {
	    endpointId: string;
	    baseUrl: string;
	    configured: boolean;
	    keySource: string;
	
	    static createFrom(source: any = {}) {
	        return new RunPodConfig(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.endpointId = source["endpointId"];
	        this.baseUrl = source["baseUrl"];
	        this.configured = source["configured"];
	        this.keySource = source["keySource"];
	    }
	}
	export class RunPodStatus {
	    ok: boolean;
	    endpointId: string;
	    message: string;
	    workers: number;
	
	    static createFrom(source: any = {}) {
	        return new RunPodStatus(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.ok = source["ok"];
	        this.endpointId = source["endpointId"];
	        this.message = source["message"];
	        this.workers = source["workers"];
	    }
	}
	
	export class SystemInfo {
	    platform: string;
	    workspace: string;
	    jobs: number;
	    python: boolean;
	
	    static createFrom(source: any = {}) {
	        return new SystemInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.platform = source["platform"];
	        this.workspace = source["workspace"];
	        this.jobs = source["jobs"];
	        this.python = source["python"];
	    }
	}

}

