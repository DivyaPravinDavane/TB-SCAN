/**
 * platform.js - TB-Scan AI Clinical Diagnostic & Triage Platform
 */

// Application State
const appState = {
  currentScan: null,
  activeRole: 'radiologist',
  selectedFile: null,
  activeSampleId: 'tb_positive',
  viewerMode: 'overlay',
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  isInverted: false,
  heatmapOpacity: 0.65
};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  setupDragAndDrop();
  setupPanAndZoom();
  setupFeedbackRadios();
  fetchAnalyticsMetrics();

  // If an active custom uploaded scan exists in sessionStorage, auto-run on it!
  const savedB64 = sessionStorage.getItem('tb_active_scan_b64');
  const savedName = sessionStorage.getItem('tb_active_scan_name');
  syncModuleActiveScanBadge('activeScanBadgeWorkspace', loadBenchmarkSample);

  if (savedB64 && document.getElementById('baseImage')) {
    appState.activeSampleId = null;
    appState.selectedFile = null;
    document.getElementById('baseImage').src = savedB64;
    document.getElementById('dualRawImage').src = savedB64;
    const dropTitle = document.querySelector('.dropzone-title');
    if (dropTitle && savedName) dropTitle.textContent = "Uploaded: " + savedName;
    triggerInference();
  } else if (document.getElementById('baseImage')) {
    // Auto-run benchmark positive sample to provide instant rich interactive view
    loadBenchmarkSample('tb_positive');
  }
});

/**
 * Tab Navigation
 */
function switchTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));

  if (tabId === 'diagnostic') {
    document.getElementById('tabBtnDiagnostic').classList.add('active');
    document.getElementById('viewDiagnostic').classList.add('active');
  } else if (tabId === 'analytics') {
    document.getElementById('tabBtnAnalytics').classList.add('active');
    document.getElementById('viewAnalytics').classList.add('active');
    fetchAnalyticsMetrics();
  }
}

/**
 * Role Context Updates
 */
function updateRoleContext(role) {
  appState.activeRole = role;
  const docSection = document.getElementById('doctorSection');
  if (role === 'technician') {
    docSection.style.opacity = '0.5';
    docSection.style.pointerEvents = 'none';
  } else {
    docSection.style.opacity = '1';
    docSection.style.pointerEvents = 'auto';
  }
}

/**
 * Drag and Drop & File Selection
 */
function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFile(dt.files[0]);
    }
  });
}

function handleFileSelected(event) {
  if (event.target.files && event.target.files[0]) {
    const file = event.target.files[0];
    event.target.value = '';
    handleFile(file);
  }
}

function handleFile(file) {
  appState.selectedFile = file;
  appState.activeSampleId = null;

  // Local preview & store in sessionStorage for all DL modules
  const reader = new FileReader();
  reader.onload = (e) => {
    const b64 = e.target.result;
    try {
      sessionStorage.setItem('tb_active_scan_b64', b64);
      sessionStorage.setItem('tb_active_scan_name', file.name);
    } catch (quotaErr) {
      console.warn("SessionStorage quota reached:", quotaErr);
    }

    if (document.getElementById('baseImage')) document.getElementById('baseImage').src = b64;
    if (document.getElementById('dualRawImage')) document.getElementById('dualRawImage').src = b64;
    if (document.getElementById('heatmapImage')) document.getElementById('heatmapImage').style.display = 'none';
    if (document.getElementById('peakPin')) document.getElementById('peakPin').style.display = 'none';
    if (document.getElementById('heatmapLegend')) document.getElementById('heatmapLegend').style.display = 'none';

    // Immediately trigger inference pipeline on the newly uploaded image!
    triggerInference();
  };
  reader.readAsDataURL(file);

  const dropTitle = document.querySelector('.dropzone-title');
  if (dropTitle) dropTitle.textContent = "Uploaded: " + file.name;
}

/**
 * Quick Clinical Benchmark Sample Loader
 */
function loadBenchmarkSample(sampleId) {
  appState.activeSampleId = sampleId;
  appState.selectedFile = null;
  // Clear custom uploaded scan from session storage and server when deliberately switching to benchmark
  sessionStorage.removeItem('tb_active_scan_b64');
  sessionStorage.removeItem('tb_active_scan_name');
  try { fetch('/api/v1/scans/reset-active', { method: 'POST' }); } catch (e) {}
  syncModuleActiveScanBadge('activeScanBadgeWorkspace', loadBenchmarkSample);

  const samplePath = sampleId === 'tb_positive'
    ? '/static/samples/sample_tb_positive.jpg'
    : '/static/samples/sample_tb_negative.jpg';

  if (document.getElementById('baseImage')) document.getElementById('baseImage').src = samplePath;
  if (document.getElementById('dualRawImage')) document.getElementById('dualRawImage').src = samplePath;
  
  const dropTitle = document.querySelector('.dropzone-title');
  if (dropTitle) {
    dropTitle.textContent = sampleId === 'tb_positive' 
      ? 'Sample: Active TB (Apical)' 
      : 'Sample: Normal CXR';
  }

  triggerInference();
}

/**
 * Trigger ML Inference & Grad-CAM
 */
async function triggerInference() {
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) loadingOverlay.style.display = 'flex';

  const formData = new FormData();
  if (appState.selectedFile) {
    formData.append('image', appState.selectedFile);
  } else if (appState.activeSampleId) {
    formData.append('sample_id', appState.activeSampleId);
  } else {
    const savedB64 = sessionStorage.getItem('tb_active_scan_b64');
    if (savedB64) {
      formData.append('image_b64', savedB64);
    } else {
      formData.append('sample_id', 'tb_positive');
    }
  }

  const ageInput = document.getElementById('inputAge');
  const sexInput = document.getElementById('inputSex');
  formData.append('patient_age', (ageInput && ageInput.value) || 45);
  formData.append('patient_sex', (sexInput && sexInput.value) || 'Male');

  try {
    const res = await fetch('/api/v1/scans/infer', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (data.status === 'success' && data.scan) {
      applyScanResults(data.scan);
    } else {
      alert("Inference failed: " + (data.message || data.error));
    }
  } catch (err) {
    console.error("Inference Error:", err);
    alert("Connection error during model inference: " + err.message);
  } finally {
    if (loadingOverlay) loadingOverlay.style.display = 'none';
  }
}

/**
 * Update UI with Diagnostic Scan Telemetry
 */
function applyScanResults(scan) {
  appState.currentScan = scan;

  // Study UID
  document.getElementById('displayStudyUid').textContent = scan.study_uid;

  // Images
  const baseImg = document.getElementById('baseImage');
  const heatImg = document.getElementById('heatmapImage');
  const dualRawImg = document.getElementById('dualRawImage');
  const dualOverlayImg = document.getElementById('dualOverlayImage');

  baseImg.src = scan.raw_image_url;
  heatImg.src = scan.heatmap_url;
  heatImg.style.display = 'block';
  heatImg.style.opacity = appState.heatmapOpacity;

  dualRawImg.src = scan.raw_image_url;
  dualOverlayImg.src = scan.overlay_url;

  // Peak activation pin
  const peakPin = document.getElementById('peakPin');
  if (scan.classification === 'HIGH_RISK') {
    peakPin.style.display = 'flex';
    // Position pin based on rel coords (e.g. 52% x, 40% y)
    const [relX, relY] = [0.52, 0.38]; // default or from metadata
    peakPin.style.left = `${relX * 100}%`;
    peakPin.style.top = `${relY * 100}%`;
    document.getElementById('pinLabel').textContent = scan.peak_region;
  } else {
    peakPin.style.display = 'none';
  }

  document.getElementById('heatmapLegend').style.display = 'flex';

  // Pipeline Flow Banner Update
  const flowTitle = document.getElementById('flowStatusTitle');
  const flowSub = document.getElementById('flowStatusSub');
  if (flowTitle && flowSub) {
    const isCustom = !!sessionStorage.getItem('tb_active_scan_name');
    const nameStr = isCustom ? sessionStorage.getItem('tb_active_scan_name') : (scan.study_uid || 'Benchmark CXR');
    flowTitle.textContent = `DL Pipeline Active: ${nameStr} • ${scan.class_label} (${(scan.tb_probability * 100).toFixed(1)}%)`;
    flowSub.textContent = `All deep learning modules are evaluated on this radiograph. Click any module below to inspect findings:`;
    document.querySelectorAll('.flow-step').forEach(s => s.classList.add('passed'));
  }

  // Score Card
  const riskBadge = document.getElementById('riskBadge');
  const probNum = document.getElementById('probNum');
  const probBar = document.getElementById('probBar');

  riskBadge.className = 'risk-badge';
  if (scan.classification === 'HIGH_RISK') {
    riskBadge.classList.add('high');
    riskBadge.textContent = 'HIGH RISK (TB Suspected)';
  } else if (scan.classification === 'NORMAL') {
    riskBadge.classList.add('normal');
    riskBadge.textContent = 'LOW RISK / NORMAL';
  } else {
    riskBadge.classList.add('indeterminate');
    riskBadge.textContent = 'INDETERMINATE / BORDERLINE';
  }

  probNum.textContent = `${(scan.tb_probability * 100).toFixed(1)}%`;
  probBar.style.width = `${Math.min(scan.tb_probability * 100, 100)}%`;

  // Anatomical Findings
  document.getElementById('valPeakRegion').textContent = scan.peak_region;
  document.getElementById('valInvolvement').textContent = scan.involvement;
  document.getElementById('valCavity').textContent = scan.cavity;
  document.getElementById('valZones').textContent = scan.zones;

  // Triage Card
  const triageCard = document.getElementById('triageCard');
  triageCard.style.display = 'block';
  document.getElementById('triageActionText').textContent = scan.triage_action;
  document.getElementById('triageTime').textContent = `${scan.latency_seconds || 1.2}s`;

  // Populate Clinical Report Modal
  document.getElementById('rptDate').textContent = scan.timestamp;
  document.getElementById('rptStudyUid').textContent = scan.study_uid;
  document.getElementById('rptAge').textContent = `${scan.patient_age} Yrs`;
  document.getElementById('rptSex').textContent = scan.patient_sex;
  document.getElementById('rptRawImg').src = scan.raw_image_url;
  document.getElementById('rptOverlayImg').src = scan.overlay_url;
  document.getElementById('rptRiskTag').textContent = scan.class_label;
  document.getElementById('rptProb').textContent = `${(scan.tb_probability * 100).toFixed(1)}%`;
  document.getElementById('rptPeak').textContent = scan.peak_region;
  document.getElementById('rptInvolvement').textContent = scan.involvement;
  document.getElementById('rptCavity').textContent = scan.cavity;
  document.getElementById('rptZones').textContent = scan.zones;

  // Update default doctor notes
  if (scan.classification === 'HIGH_RISK') {
    document.getElementById('doctorNotes').value = 
      `Significant ${scan.peak_region.toLowerCase()} with ${scan.involvement.toLowerCase()}. Compatible with active pulmonary tuberculosis. Recommend urgent sputum GeneXpert MTB/RIF test and respiratory isolation.`;
  } else {
    document.getElementById('doctorNotes').value = 
      `Clear bilateral lung fields. Normal bronchovascular markings with no active infiltrates, cavitations, or pleural effusion. Routine screening protocol.`;
  }

  fetchAnalyticsMetrics();
}

/**
 * View Mode Switcher: Overlay vs Dual-Pane
 */
function setViewerMode(mode) {
  appState.viewerMode = mode;
  const overlayViewer = document.getElementById('overlayViewer');
  const dualViewer = document.getElementById('dualViewer');
  const btnOverlay = document.getElementById('btnModeOverlay');
  const btnDual = document.getElementById('btnModeDual');

  if (mode === 'overlay') {
    overlayViewer.style.display = 'flex';
    dualViewer.style.display = 'none';
    btnOverlay.classList.add('active');
    btnDual.classList.remove('active');
  } else {
    overlayViewer.style.display = 'none';
    dualViewer.style.display = 'grid';
    btnOverlay.classList.remove('active');
    btnDual.classList.add('active');
  }
}

/**
 * Opacity Slider
 */
function updateOpacity(val) {
  const op = val / 100;
  appState.heatmapOpacity = op;
  document.getElementById('opacityVal').textContent = `${val}%`;
  const heatImg = document.getElementById('heatmapImage');
  if (heatImg) heatImg.style.opacity = op;
}

/**
 * Pan & Zoom Controls
 */
function setupPanAndZoom() {
  const container = document.getElementById('viewportContainer');
  const stage = document.getElementById('imageStage');

  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    adjustZoom(delta);
  });

  stage.addEventListener('mousedown', (e) => {
    appState.isDragging = true;
    appState.dragStartX = e.clientX - appState.panX;
    appState.dragStartY = e.clientY - appState.panY;
    stage.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!appState.isDragging) return;
    appState.panX = e.clientX - appState.dragStartX;
    appState.panY = e.clientY - appState.dragStartY;
    updateStageTransform();
  });

  window.addEventListener('mouseup', () => {
    appState.isDragging = false;
    stage.style.cursor = 'grab';
  });
}

function adjustZoom(delta) {
  appState.zoom = Math.max(0.5, Math.min(3.5, appState.zoom + delta));
  updateStageTransform();
}

function updateStageTransform() {
  const stage = document.getElementById('imageStage');
  stage.style.transform = `translate(${appState.panX}px, ${appState.panY}px) scale(${appState.zoom})`;
}

function toggleInvert() {
  appState.isInverted = !appState.isInverted;
  const baseImg = document.getElementById('baseImage');
  const dualRaw = document.getElementById('dualRawImage');
  const filterVal = appState.isInverted ? 'invert(1) contrast(1.2)' : 'none';
  baseImg.style.filter = filterVal;
  dualRaw.style.filter = filterVal;

  const btn = document.getElementById('btnInvert');
  if (appState.isInverted) {
    btn.style.background = 'rgba(0, 229, 255, 0.2)';
    btn.style.color = '#00e5ff';
  } else {
    btn.style.background = 'rgba(255, 255, 255, 0.05)';
    btn.style.color = 'inherit';
  }
}

function resetViewTransforms() {
  appState.zoom = 1.0;
  appState.panX = 0;
  appState.panY = 0;
  updateStageTransform();
  if (appState.isInverted) toggleInvert();
}

/**
 * Radio Feedback Buttons
 */
function setupFeedbackRadios() {
  document.querySelectorAll('.feedback-btn').forEach(label => {
    label.addEventListener('click', () => {
      document.querySelectorAll('.feedback-btn').forEach(l => l.classList.remove('active'));
      label.classList.add('active');
    });
  });
}

/**
 * Submit Electronic Sign-off
 */
async function submitSignoff() {
  if (!appState.currentScan) {
    alert("Please analyze a chest X-ray first before signing off.");
    return;
  }

  const notes = document.getElementById('doctorNotes').value;
  const reviewer = document.getElementById('reviewedBy').value;
  const feedbackInput = document.querySelector('input[name="radiologistFeedback"]:checked');
  const feedback = feedbackInput ? feedbackInput.value : 'AGREE';

  try {
    const res = await fetch(`/api/v1/scans/${appState.currentScan.id}/signoff`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doctor_notes: notes,
        reviewed_by: reviewer,
        radiologist_feedback: feedback
      })
    });

    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('rptNotes').textContent = notes;
      document.getElementById('rptDoctor').textContent = reviewer;
      alert(`Scan ${appState.currentScan.study_uid} successfully verified and signed by ${reviewer}.`);
      openReportModal();
      fetchAnalyticsMetrics();
    }
  } catch (err) {
    alert("Error saving signoff: " + err.message);
  }
}

/**
 * Clinical Report Modal
 */
function openReportModal() {
  document.getElementById('reportModal').style.display = 'flex';
}

function closeReportModal() {
  document.getElementById('reportModal').style.display = 'none';
}

/**
 * Analytics & Audit Log Fetcher
 */
async function fetchAnalyticsMetrics() {
  try {
    const res = await fetch('/api/v1/analytics/metrics');
    const data = await res.json();

    if (data.status === 'success') {
      document.getElementById('kpiTotal').textContent = data.total_scans;
      document.getElementById('kpiPositivity').textContent = `${data.positivity_rate}%`;
      document.getElementById('kpiNormal').textContent = data.normal_count;
      document.getElementById('kpiLatency').textContent = data.avg_turnaround_time;

      renderAuditTable(data.scans || []);
    }
  } catch (err) {
    console.error("Failed to fetch analytics metrics:", err);
  }
}

function renderAuditTable(scans) {
  const tbody = document.getElementById('auditTableBody');
  tbody.innerHTML = '';

  scans.forEach(scan => {
    const tr = document.createElement('tr');
    const riskBadgeClass = scan.classification === 'HIGH_RISK' ? 'color:#f87171;' : scan.classification === 'NORMAL' ? 'color:#34d399;' : 'color:#fbbf24;';
    const riskLabel = scan.classification === 'HIGH_RISK' ? 'High Risk' : scan.classification === 'NORMAL' ? 'Normal' : 'Borderline';
    const reviewerText = scan.reviewed_by ? scan.reviewed_by : '<span style="color:#64748b;">Pending Review</span>';

    tr.innerHTML = `
      <td><strong>${scan.study_uid}</strong></td>
      <td style="color:#94a3b8; font-size:11.5px;">${scan.timestamp}</td>
      <td><span style="font-weight:700; ${riskBadgeClass}">● ${riskLabel}</span></td>
      <td>${(scan.tb_probability * 100).toFixed(1)}%</td>
      <td>${scan.peak_region || 'N/A'}</td>
      <td><span style="font-size:11px; padding:2px 8px; border-radius:4px; background:rgba(255,255,255,0.06);">${scan.status}</span></td>
      <td style="font-size:11.5px;">${reviewerText}</td>
      <td>
        <button class="btn-secondary btn-sm" onclick="switchTab('diagnostic'); loadBenchmarkSample('${scan.classification === 'HIGH_RISK' ? 'tb_positive' : 'tb_negative'}')">
          View Scan
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAuditTable(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('#auditTableBody tr').forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

/**
 * Navbar Notifications Dropdown Toggle
 */
function toggleNotifDropdown() {
  const dd = document.getElementById('notifDropdown');
  if (!dd) return;
  const isShown = dd.style.display === 'block';
  dd.style.display = isShown ? 'none' : 'block';
}

document.addEventListener('click', (e) => {
  const notifWrapper = document.querySelector('.triage-notif-wrapper') || document.querySelector('.notification-wrapper');
  const notifDropdown = document.getElementById('notifDropdown');
  if (notifWrapper && notifDropdown && !notifWrapper.contains(e.target)) {
    notifDropdown.style.display = 'none';
  }
});

/**
 * Intelligent Omni-Search Execution
 */
function executeOmniSearch(query) {
  if (!query || !query.trim()) return;
  const q = query.trim().toLowerCase();

  // Smart Module Routing
  if (q.includes('qa') || q.includes('input') || q.includes('artifact') || q.includes('exposure') || q.includes('motion')) {
    window.location.href = '/input-qa';
    return;
  }
  if (q.includes('seg') || q.includes('lung') || q.includes('bone') || q.includes('rib') || q.includes('u-net')) {
    window.location.href = '/segmentation';
    return;
  }
  if (q.includes('diff') || q.includes('hallmark') || q.includes('cavit') || q.includes('infiltrate') || q.includes('pneumonia')) {
    window.location.href = '/differential';
    return;
  }
  if (q.includes('xai') || q.includes('grad') || q.includes('cam') || q.includes('yolo') || q.includes('box') || q.includes('heat')) {
    window.location.href = '/explainability';
    return;
  }
  if (q.includes('treat') || q.includes('improv') || q.includes('recover') || q.includes('cure') || q.includes('clearance')) {
    window.location.href = '/treatment-analytics';
    return;
  }
  if (q.includes('dots') || q.includes('longitudinal') || q.includes('delta') || q.includes('tracking')) {
    window.location.href = '/longitudinal';
    return;
  }
  if (q.includes('unc') || q.includes('monte') || q.includes('dropout') || q.includes('variance') || q.includes('triage')) {
    window.location.href = '/uncertainty';
    return;
  }
  if (q.includes('analytic') || q.includes('surveillance') || q.includes('registry') || q.includes('kpi') || q.includes('audit')) {
    window.location.href = '/analytics';
    return;
  }
  if (q.includes('work') || q.includes('home') || q.includes('scan') || q.includes('detect')) {
    window.location.href = '/';
    return;
  }

  // If already on analytics page, filter the audit table directly
  const searchInput = document.getElementById('tableSearch');
  if (searchInput && typeof filterAuditTable === 'function') {
    searchInput.value = query;
    filterAuditTable(query);
  } else {
    // Navigate to analytics with search query
    window.location.href = `/analytics?search=${encodeURIComponent(query)}`;
  }
}

// Global Keyboard Shortcut: ⌘K or Ctrl+K
window.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    const omni = document.getElementById('omniSearchInput');
    if (omni) {
      omni.focus();
      omni.select();
    }
  }
});

/**
 * PACS Modal Handlers
 */
function openPacsModal() {
  const m = document.getElementById('pacsModal');
  if (m) m.style.display = 'flex';
}

function closePacsModal() {
  const m = document.getElementById('pacsModal');
  if (m) m.style.display = 'none';
}

/**
 * WHO Clinical Protocol Modal Handlers
 */
function openProtocolModal() {
  const m = document.getElementById('protocolModal');
  if (m) m.style.display = 'flex';
}

function closeProtocolModal() {
  const m = document.getElementById('protocolModal');
  if (m) m.style.display = 'none';
}

/**
 * Quick Modules Flyout Menu Handler
 */
function toggleModulesMenu() {
  const flyout = document.getElementById('modulesFlyout');
  if (!flyout) return;
  const isShown = flyout.style.display === 'block';
  flyout.style.display = isShown ? 'none' : 'block';
}

document.addEventListener('click', (e) => {
  const wrapper = document.querySelector('.modules-dropdown-wrapper');
  const flyout = document.getElementById('modulesFlyout');
  if (wrapper && flyout && !wrapper.contains(e.target)) {
    flyout.style.display = 'none';
  }
});

/**
 * Universal DL Module Cross-Page Helpers
 */

/**
 * Creates FormData containing user uploaded scan (via image_b64) or falls back to server active image
 */
function getActiveScanFormData(fallbackSampleId = null) {
  const formData = new FormData();
  const savedB64 = sessionStorage.getItem('tb_active_scan_b64');
  if (savedB64) {
    formData.append('image_b64', savedB64);
  } else if (fallbackSampleId) {
    formData.append('sample_id', fallbackSampleId);
  }
  return formData;
}

/**
 * Handles file upload from within any specialized DL module page
 */
function handleModuleFileUpload(file, onLoadedCallback) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const b64 = e.target.result;
    try {
      sessionStorage.setItem('tb_active_scan_b64', b64);
      sessionStorage.setItem('tb_active_scan_name', file.name);
    } catch (quotaErr) {
      console.warn("SessionStorage limit reached, falling back to server active cache:", quotaErr);
    }
    if (typeof onLoadedCallback === 'function') {
      onLoadedCallback(file, b64);
    }
  };
  reader.readAsDataURL(file);
}

/**
 * Synchronizes the active scan badge across module sidebars with server state
 */
async function syncModuleActiveScanBadge(containerId, onResetCallback) {
  const container = document.getElementById(containerId);
  if (!container) return;
  let savedName = sessionStorage.getItem('tb_active_scan_name');
  
  if (!savedName) {
    try {
      const res = await fetch('/api/v1/scans/active-status');
      const data = await res.json();
      if (data.status === 'success' && data.has_active_scan) {
        savedName = data.filename || 'Custom Ingested Radiograph';
        try { sessionStorage.setItem('tb_active_scan_name', savedName); } catch (e) {}
      }
    } catch (e) {
      console.error(e);
    }
  }

  if (savedName) {
    container.innerHTML = `
      <div class="active-scan-chip" style="display:flex; align-items:center; justify-content:space-between; padding:8px 12px; background:rgba(0, 229, 255, 0.09); border:1px solid rgba(0, 229, 255, 0.3); border-radius:8px; margin-bottom:12px; font-size:11.5px;">
        <div style="display:flex; align-items:center; gap:8px; overflow:hidden;">
          <span style="width:8px; height:8px; border-radius:50%; background:#00e5ff; box-shadow:0 0 8px #00e5ff; flex-shrink:0;"></span>
          <span style="color:#e2e8f0; font-weight:600; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">Active CXR: <strong>${savedName}</strong></span>
        </div>
        <button type="button" style="background:transparent; border:none; color:#38bdf8; cursor:pointer; font-size:11px; font-weight:700; text-decoration:underline; margin-left:8px; flex-shrink:0;" onclick="clearUploadedScanAndReset(${onResetCallback ? onResetCallback.name : 'null'})">Use Benchmark</button>
      </div>
    `;
    container.style.display = 'block';
  } else {
    container.innerHTML = '';
    container.style.display = 'none';
  }
}

/**
 * Clears uploaded scan on both client and server, resetting module to default benchmark
 */
async function clearUploadedScanAndReset(callback) {
  sessionStorage.removeItem('tb_active_scan_b64');
  sessionStorage.removeItem('tb_active_scan_name');
  try {
    await fetch('/api/v1/scans/reset-active', { method: 'POST' });
  } catch (e) {}
  if (typeof callback === 'function') {
    callback('tb_positive');
  } else {
    location.reload();
  }
}

