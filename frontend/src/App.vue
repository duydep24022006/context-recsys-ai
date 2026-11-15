<template>
  <div>
    <header class="topbar">
      <div class="brand">
        <h1>Hệ thống Khuyến nghị Sản phẩm Thương mại Điện tử theo Ngữ cảnh</h1>
        <p>
          MLOps Real-time Recommender · Model:
          <b>{{ modelInfo.model_name_tag || modelInfo.name || 'ContextAwareRecommender' }}</b>
          <span v-if="modelInfo.version"> (v{{ modelInfo.version }} · @{{ modelInfo.alias || 'Production' }})</span>
        </p>
      </div>
      <div class="top-actions">
        <span class="status-pill">
          ● {{ health.model_loaded ? 'Production Ready' : 'Fallback Mode' }}
        </span>
        <button class="secondary" @click="toggleTheme" :aria-label="'Đổi giao diện Sáng/Tối'">
          {{ dark ? '☀ Chế độ Sáng' : '🌙 Chế độ Tối' }}
        </button>
      </div>
    </header>

    <nav class="nav-tabs" role="tablist" aria-label="Điều hướng bảng điều khiển">
      <button
        v-for="t in tabs"
        :key="t.id"
        role="tab"
        :aria-selected="activeTab === t.id"
        :class="['tab-btn', { active: activeTab === t.id }]"
        @click="activeTab = t.id"
      >
        {{ t.label }}
      </button>
    </nav>

    <main class="container">
      <!-- TAB 1: GỢI Ý REAL-TIME & SO SÁNH NGỮ CẢNH -->
      <section v-if="activeTab === 'recommend'" aria-label="Gợi ý sản phẩm thời gian thực">
        <div class="card">
          <h2>
            <span>1. Truy vấn Gợi ý theo Ngữ cảnh Người dùng</span>
            <span v-if="userProfile" style="font-size: 13px; font-weight: 500; color: var(--mute)">
              Lịch sử User #{{ form.user_id }}:
              <b>{{ userProfile.total_interactions }}</b> tương tác
              <span v-if="userProfile.cold_start" class="tag Accessories" style="margin-left: 6px">Cold-Start User</span>
              <span v-else-if="userProfile.favorite_categories?.length" style="margin-left: 6px">
                · Ưa thích:
                <span
                  v-for="fc in userProfile.favorite_categories.slice(0, 3)"
                  :key="fc.category"
                  :class="['tag', fc.category]"
                  style="margin-right: 4px"
                >
                  {{ fc.category }} ({{ (fc.ratio * 100).toFixed(0) }}%)
                </span>
              </span>
            </span>
          </h2>

          <form class="controls" @submit.prevent="runRecommendAndCompare">
            <label>
              User ID (1–500, &gt;500 = Cold-Start)
              <input type="number" min="1" max="999999" v-model.number="form.user_id" />
            </label>
            <label>
              Thiết bị (Ngữ cảnh A)
              <select v-model="form.device">
                <option value="mobile">Mobile (Điện thoại)</option>
                <option value="desktop">Desktop (Máy tính)</option>
              </select>
            </label>
            <label>
              Giờ truy cập A (0–23h)
              <input type="number" min="0" max="23" v-model.number="form.hour" />
            </label>
            <label>
              Thiết bị đối chiếu (Ngữ cảnh B)
              <select v-model="compareForm.device">
                <option value="desktop">Desktop (Máy tính)</option>
                <option value="mobile">Mobile (Điện thoại)</option>
              </select>
            </label>
            <label>
              Giờ đối chiếu B (0–23h)
              <input type="number" min="0" max="23" v-model.number="compareForm.hour" />
            </label>
            <label>
              Top-K
              <input type="number" min="1" max="50" v-model.number="form.top_k" />
            </label>
            <button type="submit" class="primary" :disabled="loadingRec">
              {{ loadingRec ? 'Đang suy luận...' : 'Lấy gợi ý & So sánh 2 ngữ cảnh' }}
            </button>
          </form>

          <div v-if="recError" class="error-box">{{ recError }}</div>
        </div>

        <div class="grid-2" v-if="recResultA || recResultB">
          <!-- Context A -->
          <div class="card">
            <h2>
              <span>Ngữ cảnh A: {{ form.device.toUpperCase() }} · {{ form.hour }}:00h</span>
              <span v-if="recResultA" class="status-pill">{{ recResultA.latency_ms }} ms</span>
            </h2>
            <div v-if="!recResultA" class="empty-box">Chưa có dữ liệu</div>
            <div v-else class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Mã SP</th>
                    <th>Tên sản phẩm</th>
                    <th>Danh mục</th>
                    <th>Giá ($)</th>
                    <th>Điểm (Score)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(it, idx) in recResultA.items" :key="it.product_id">
                    <td class="num">{{ idx + 1 }}</td>
                    <td class="num">#{{ it.product_id }}</td>
                    <td>{{ it.title }}</td>
                    <td><span :class="['tag', it.category]">{{ it.category }}</span></td>
                    <td class="num">${{ it.price.toFixed(2) }}</td>
                    <td class="num"><b>{{ (it.score ?? 0).toFixed(4) }}</b></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Context B -->
          <div class="card">
            <h2>
              <span>Ngữ cảnh B: {{ compareForm.device.toUpperCase() }} · {{ compareForm.hour }}:00h</span>
              <span v-if="recResultB" class="status-pill">{{ recResultB.latency_ms }} ms</span>
            </h2>
            <div v-if="!recResultB" class="empty-box">Chưa có dữ liệu</div>
            <div v-else class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Mã SP</th>
                    <th>Tên sản phẩm</th>
                    <th>Danh mục</th>
                    <th>Giá ($)</th>
                    <th>Điểm (Score)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(it, idx) in recResultB.items" :key="it.product_id">
                    <td class="num">{{ idx + 1 }}</td>
                    <td class="num">#{{ it.product_id }}</td>
                    <td>{{ it.title }}</td>
                    <td><span :class="['tag', it.category]">{{ it.category }}</span></td>
                    <td class="num">${{ it.price.toFixed(2) }}</td>
                    <td class="num"><b>{{ (it.score ?? 0).toFixed(4) }}</b></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <!-- TAB 2: GIAO DỊCH GẦN ĐÂY -->
      <section v-if="activeTab === 'transactions'" aria-label="Lịch sử giao dịch gần đây">
        <div class="card">
          <h2>
            <span>Luồng Sự kiện &amp; Giao dịch Gần đây (Tự động làm mới 10s)</span>
            <button class="secondary" @click="fetchTransactions">Làm mới ngay</button>
          </h2>
          <div class="controls">
            <label>
              Lọc theo User ID
              <input type="number" min="1" placeholder="Tất cả" v-model.number="txFilter.user_id" @change="fetchTransactions" />
            </label>
            <label>
              Hành vi (Event)
              <select v-model="txFilter.event_type" @change="fetchTransactions">
                <option value="">Tất cả hành vi</option>
                <option value="view">view (Xem)</option>
                <option value="cart">cart (Thêm giỏ)</option>
                <option value="purchase">purchase (Mua hàng)</option>
              </select>
            </label>
            <label>
              Thiết bị (Device)
              <select v-model="txFilter.device_type" @change="fetchTransactions">
                <option value="">Tất cả thiết bị</option>
                <option value="mobile">mobile</option>
                <option value="desktop">desktop</option>
              </select>
            </label>
            <label>
              Số dòng hiển thị
              <select v-model.number="txFilter.limit" @change="fetchTransactions">
                <option :value="20">20 dòng</option>
                <option :value="50">50 dòng</option>
                <option :value="100">100 dòng</option>
              </select>
            </label>
          </div>

          <div v-if="recentTx.length === 0" class="empty-box">Không có giao dịch thỏa bộ lọc.</div>
          <div v-else class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Thời gian</th>
                  <th>User ID</th>
                  <th>Mã SP</th>
                  <th>Sản phẩm</th>
                  <th>Danh mục</th>
                  <th>Hành vi</th>
                  <th>Thiết bị</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(r, idx) in recentTx" :key="idx">
                  <td class="num">{{ String(r.timestamp).slice(0, 19) }}</td>
                  <td class="num">#{{ r.user_id }}</td>
                  <td class="num">#{{ r.product_id }}</td>
                  <td>{{ r.title }}</td>
                  <td><span :class="['tag', r.category]">{{ r.category }}</span></td>
                  <td><b>{{ r.event_type }}</b></td>
                  <td>{{ r.device_type }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- TAB 3: PHÂN TÍCH NGỮ CẢNH & HIỆU NĂNG -->
      <section v-if="activeTab === 'analytics'" aria-label="Phân tích dữ liệu và độ trễ">
        <div class="grid-4">
          <div class="kpi-card">
            <div class="kpi-label">Tổng tương tác</div>
            <div class="kpi-value">{{ analytics.total_interactions?.toLocaleString() || 0 }}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Số người dùng</div>
            <div class="kpi-value">{{ analytics.unique_users || 0 }}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Số sản phẩm</div>
            <div class="kpi-value">{{ analytics.unique_products || 0 }}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Độ trễ trung bình gần nhất</div>
            <div class="kpi-value">{{ avgRecentLatency }} ms</div>
          </div>
        </div>

        <div class="grid-2">
          <div class="card">
            <h2>Phân bố Danh mục trên Mobile</h2>
            <div v-for="(cnt, cat) in analytics.category_by_device?.mobile || {}" :key="cat" class="bar-row">
              <span class="bar-label">{{ cat }}</span>
              <div class="bar-track">
                <div class="bar-fill" :style="{ width: pct(cnt, mobileTotal) + '%' }"></div>
              </div>
              <span class="num" style="width: 90px; text-align: right">
                {{ cnt }} ({{ pct(cnt, mobileTotal) }}%)
              </span>
            </div>
          </div>

          <div class="card">
            <h2>Phân bố Danh mục trên Desktop</h2>
            <div v-for="(cnt, cat) in analytics.category_by_device?.desktop || {}" :key="cat" class="bar-row">
              <span class="bar-label">{{ cat }}</span>
              <div class="bar-track">
                <div class="bar-fill" :style="{ width: pct(cnt, desktopTotal) + '%' }"></div>
              </div>
              <span class="num" style="width: 90px; text-align: right">
                {{ cnt }} ({{ pct(cnt, desktopTotal) }}%)
              </span>
            </div>
          </div>
        </div>

        <div class="grid-2">
          <div class="card">
            <h2>Lưu lượng Tương tác theo Khung Giờ (0–23h)</h2>
            <div class="table-wrap" style="max-height: 290px">
              <table>
                <thead>
                  <tr>
                    <th>Giờ</th>
                    <th>Mobile</th>
                    <th>Desktop</th>
                    <th>Tổng</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="h in analytics.hourly_by_device || []" :key="h.hour">
                    <td class="num">{{ String(h.hour).padStart(2, '0') }}:00</td>
                    <td class="num">{{ h.mobile || 0 }}</td>
                    <td class="num">{{ h.desktop || 0 }}</td>
                    <td class="num"><b>{{ (h.mobile || 0) + (h.desktop || 0) }}</b></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <h2>Lịch sử Độ trễ Suy luận (Real-time Requests)</h2>
            <div v-if="latencyLog.length === 0" class="empty-box">Chưa có truy vấn gợi ý nào.</div>
            <div v-else>
              <div v-for="(entry, i) in latencyLog.slice(0, 10)" :key="i" class="bar-row">
                <span class="bar-label">{{ entry.ctx }}</span>
                <div class="bar-track">
                  <div class="bar-fill" :style="{ width: Math.min(100, entry.ms * 2) + '%' }"></div>
                </div>
                <span class="num" style="width: 75px; text-align: right">{{ entry.ms }} ms</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- TAB 4: MÔ HÌNH & MLOPS REGISTRY -->
      <section v-if="activeTab === 'models'" aria-label="Quản lý mô hình và MLOps">
        <div class="grid-2">
          <div class="card">
            <h2>
              <span>Model Đang Phục vụ (@Production)</span>
              <button class="primary" @click="triggerReload" :disabled="reloading">
                {{ reloading ? 'Đang nạp...' : '↻ Nạp lại Model (/admin/reload)' }}
              </button>
            </h2>
            <table>
              <tbody>
                <tr><th>Tên đăng ký (Registry)</th><td><code>{{ modelInfo.name }}</code></td></tr>
                <tr><th>Phiên bản (Version)</th><td><b>v{{ modelInfo.version || 'N/A' }}</b> (@{{ modelInfo.alias || 'Production' }})</td></tr>
                <tr><th>Thuật toán / Run</th><td>{{ modelInfo.model_name_tag || 'LightGBM_Context_Aware' }}</td></tr>
                <tr><th>MLflow Run ID</th><td><code>{{ modelInfo.run_id || 'N/A' }}</code></td></tr>
                <tr><th>Thời điểm nạp</th><td>{{ modelInfo.loaded_at || 'N/A' }}</td></tr>
                <tr v-if="modelInfo.metrics?.ndcg_at_10">
                  <th>NDCG@10 / ROC-AUC</th>
                  <td class="num"><b>{{ modelInfo.metrics.ndcg_at_10 }}</b> / {{ modelInfo.metrics.auc }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="card">
            <h2>Lần Retrain Tự động Gần nhất (Prefect Pipeline)</h2>
            <div v-if="!modelInfo.last_retrain" class="empty-box">
              Chưa có lịch sử retrain hoặc đang dùng model từ lần chạy Benchmark đầu tiên.
            </div>
            <table v-else>
              <tbody>
                <tr>
                  <th>Quyết định Promote</th>
                  <td>
                    <span class="status-pill">
                      {{ modelInfo.last_retrain.promoted ? '✓ PROMOTED TO PRODUCTION' : '⏭ SKIPPED (GIỮ BẢN CŨ)' }}
                    </span>
                  </td>
                </tr>
                <tr><th>Version ứng viên</th><td>v{{ modelInfo.last_retrain.candidate_version }} (trước đó: v{{ modelInfo.last_retrain.previous_version || '1' }})</td></tr>
                <tr><th>NDCG@10 mới vs cũ</th><td class="num"><b>{{ modelInfo.last_retrain.new_ndcg_at_10 }}</b> vs {{ modelInfo.last_retrain.previous_ndcg_at_10 ?? 'N/A' }}</td></tr>
                <tr><th>Phát hiện Data Drift</th><td>{{ modelInfo.last_retrain.drift_detected ? 'Có (PSI vượt ngưỡng)' : 'Ổn định' }}</td></tr>
                <tr><th>Lý do</th><td>{{ modelInfo.last_retrain.reason }}</td></tr>
                <tr><th>Thời gian</th><td>{{ modelInfo.last_retrain.timestamp }}</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <h2>Bảng So sánh Thực nghiệm các Mô hình (CLO1 Benchmark trên MLflow)</h2>
          <div v-if="benchmarkRows.length === 0" class="empty-box">
            Chưa có kết quả benchmark. Hãy chạy <code>make benchmark</code> hoặc <code>python src/benchmark.py</code>.
          </div>
          <div v-else class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mô hình</th>
                  <th>Context-Aware</th>
                  <th>Precision@10</th>
                  <th>Recall@10</th>
                  <th>NDCG@10</th>
                  <th>MAP@10</th>
                  <th>ROC-AUC</th>
                  <th>Suy luận (ms/query)</th>
                  <th>Huấn luyện (s)</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="m in benchmarkRows" :key="m.Model">
                  <td><b>{{ m.Model }}</b></td>
                  <td>{{ m.context_aware ? '✓ Có' : 'Không' }}</td>
                  <td class="num">{{ fmt(m.precision_at_10_mean ?? m.precision_at_10) }}</td>
                  <td class="num">{{ fmt(m.recall_at_10_mean ?? m.recall_at_10) }}</td>
                  <td class="num"><b>{{ fmt(m.ndcg_at_10_mean ?? m.ndcg_at_10) }}</b></td>
                  <td class="num">{{ fmt(m.map_at_10_mean ?? m.map_at_10) }}</td>
                  <td class="num">{{ fmt(m.auc_mean ?? m.auc) }}</td>
                  <td class="num">{{ fmt(m.inference_ms_per_query_mean ?? m.inference_ms_per_query, 2) }}</td>
                  <td class="num">{{ fmt(m.train_time_sec_mean ?? m.train_time_sec, 2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'

const tabs = [
  { id: 'recommend', label: '1. Gợi ý Real-time & So sánh Ngữ cảnh' },
  { id: 'transactions', label: '2. Giao dịch Gần đây' },
  { id: 'analytics', label: '3. Phân tích Ngữ cảnh & Độ trễ' },
  { id: 'models', label: '4. Mô hình & MLOps Registry' }
]
const activeTab = ref('recommend')
const dark = ref(false)

const health = reactive({ status: 'ok', model_loaded: false })
const modelInfo = reactive({})
const benchmarkRows = ref([])
const userProfile = ref(null)

const form = reactive({ user_id: 15, device: 'mobile', hour: 20, top_k: 10 })
const compareForm = reactive({ device: 'desktop', hour: 10 })
const recResultA = ref(null)
const recResultB = ref(null)
const loadingRec = ref(false)
const recError = ref('')
const latencyLog = ref([])

const txFilter = reactive({ user_id: null, event_type: '', device_type: '', limit: 20 })
const recentTx = ref([])
const analytics = reactive({})
const reloading = ref(false)

function toggleTheme() {
  dark.value = !dark.value
  document.documentElement.setAttribute('data-theme', dark.value ? 'dark' : 'light')
}

function fmt(val, digits = 4) {
  return typeof val === 'number' ? val.toFixed(digits) : val
}

function pct(val, total) {
  if (!total) return 0
  return Math.round((val / total) * 100)
}

const mobileTotal = computed(() => {
  const m = analytics.category_by_device?.mobile || {}
  return Object.values(m).reduce((a, b) => a + b, 0)
})

const desktopTotal = computed(() => {
  const d = analytics.category_by_device?.desktop || {}
  return Object.values(d).reduce((a, b) => a + b, 0)
})

const avgRecentLatency = computed(() => {
  if (latencyLog.value.length === 0) return '0.0'
  const sum = latencyLog.value.reduce((a, b) => a + b.ms, 0)
  return (sum / latencyLog.value.length).toFixed(2)
})

async function callRecommend(payload) {
  const r = await fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!r.ok) {
    const errBody = await r.json().catch(() => ({}))
    throw new Error(errBody.detail || r.statusText)
  }
  return await r.json()
}

async function runRecommendAndCompare() {
  loadingRec.value = true
  recError.value = ''
  try {
    const [resA, resB, prof] = await Promise.all([
      callRecommend({
        user_id: form.user_id,
        device: form.device,
        hour: form.hour,
        top_k: form.top_k
      }),
      callRecommend({
        user_id: form.user_id,
        device: compareForm.device,
        hour: compareForm.hour,
        top_k: form.top_k
      }),
      fetch(`/users/${form.user_id}/profile`).then((r) => (r.ok ? r.json() : null)).catch(() => null)
    ])
    recResultA.value = resA
    recResultB.value = resB
    userProfile.value = prof
    latencyLog.value.unshift(
      { ctx: `${form.device} ${form.hour}h (U#${form.user_id})`, ms: resA.latency_ms },
      { ctx: `${compareForm.device} ${compareForm.hour}h (U#${form.user_id})`, ms: resB.latency_ms }
    )
  } catch (e) {
    recError.value = 'Lỗi khi lấy gợi ý: ' + e.message
  } finally {
    loadingRec.value = false
  }
}

async function fetchTransactions() {
  const params = new URLSearchParams()
  params.set('limit', String(txFilter.limit || 20))
  if (txFilter.user_id) params.set('user_id', String(txFilter.user_id))
  if (txFilter.event_type) params.set('event_type', txFilter.event_type)
  if (txFilter.device_type) params.set('device_type', txFilter.device_type)
  try {
    const r = await fetch(`/interactions/recent?${params.toString()}`)
    if (r.ok) recentTx.value = await r.json()
  } catch (e) {}
}

async function fetchMetadata() {
  try {
    const [h, m, b, a] = await Promise.all([
      fetch('/health').then((r) => (r.ok ? r.json() : {})),
      fetch('/model/info').then((r) => (r.ok ? r.json() : {})),
      fetch('/models/benchmark').then((r) => (r.ok ? r.json() : [])),
      fetch('/analytics/summary').then((r) => (r.ok ? r.json() : {}))
    ])
    Object.assign(health, h)
    Object.assign(modelInfo, m)
    benchmarkRows.value = Array.isArray(b) ? b : []
    Object.assign(analytics, a)
  } catch (e) {}
}

async function triggerReload() {
  reloading.value = true
  try {
    await fetch('/admin/reload', { method: 'POST' })
    await fetchMetadata()
    await runRecommendAndCompare()
  } finally {
    reloading.value = false
  }
}

onMounted(async () => {
  await fetchMetadata()
  await fetchTransactions()
  await runRecommendAndCompare()
  setInterval(() => {
    fetchTransactions()
    fetchMetadata()
  }, 10000)
})
</script>
