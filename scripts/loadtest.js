import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '60s',
  thresholds: {
    http_req_duration: ['p(95)<2000'],
    http_req_failed: ['rate<0.05'],
  },
};

const BASE_URL = __ENV.BASE_URL || __ENV.TARGET_URL || 'http://127.0.0.1:8000';
const API_KEY = __ENV.API_KEY || '';

export default function () {
  const headers = {
    'Content-Type': 'application/json',
  };
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  // 1. Health Endpoint
  const healthRes = http.get(`${BASE_URL}/api/v1/health`, { headers });
  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
  });

  // 2. Distortion Generation Endpoint
  const dreamPayload = JSON.stringify({
    text: 'Load testing NightmareNet API endpoint for latency and stability verification.',
    strength: 0.3,
  });
  const dreamRes = http.post(`${BASE_URL}/api/v1/generate/dream`, dreamPayload, { headers });
  check(dreamRes, {
    'dream distortion status is 200': (r) => r.status === 200,
  });

  // 3. Pipeline List Endpoint
  const pipelineRes = http.get(`${BASE_URL}/api/v1/pipeline/runs`, { headers });
  check(pipelineRes, {
    'pipeline runs status is 200': (r) => r.status === 200,
  });

  sleep(0.5);
}

export function handleSummary(data) {
  return {
    'loadtest-results.json': JSON.stringify(data, null, 2),
  };
}
