const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorDiv = document.getElementById("error");

// 드래그앤드롭
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

async function uploadFile(file) {
    // UI 상태
    loading.classList.remove("hidden");
    results.classList.add("hidden");
    errorDiv.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/analyze", { method: "POST", body: formData });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        errorDiv.textContent = `Error: ${err.message}`;
        errorDiv.classList.remove("hidden");
    } finally {
        loading.classList.add("hidden");
    }
}

function renderResults(data) {
    // 어노테이션 이미지
    const img = document.getElementById("annotated-image");
    img.src = "data:image/png;base64," + data.annotated_image;

    // 각도 카드
    const cards = document.getElementById("angle-cards");
    const angles = data.angles;
    cards.innerHTML = "";

    // Cobb
    const cobb = angles.cobb;
    cards.innerHTML += `
        <div class="angle-card cobb">
            <div class="label">Cobb Angle</div>
            <div class="value">${cobb.cobb_angle}&deg;</div>
            <div class="detail">${cobb.upper_vertebra} &ndash; ${cobb.lower_vertebra}</div>
        </div>`;

    // Kyphosis
    const kyph = angles.kyphosis;
    cards.innerHTML += `
        <div class="angle-card kyphosis">
            <div class="label">Kyphosis (T1-T12)</div>
            <div class="value">${kyph.kyphosis_angle}&deg;</div>
        </div>`;

    // Lordosis
    const lord = angles.lordosis;
    cards.innerHTML += `
        <div class="angle-card lordosis">
            <div class="label">Lordosis (L1-L5)</div>
            <div class="value">${lord.lordosis_angle}&deg;</div>
        </div>`;

    // 세그먼트 테이블
    const tbody = document.querySelector("#segment-table tbody");
    tbody.innerHTML = "";
    for (const seg of angles.segments) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${seg.segment}</td><td>${seg.angle}&deg;</td>`;
        tbody.appendChild(tr);
    }

    results.classList.remove("hidden");
}
