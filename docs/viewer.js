/* Inline scenic-bubble viewer: PlayCanvas 2.20.5 (vendored, MIT) rendering
 * scene.sog. Camera lives inside the bubble at the head-box origin:
 * drag = look around, wheel/pinch = zoom (fov), WASD/arrows = move within
 * the head-box (clamped to the pipeline's +-0.5m lateral, +0.2/-1.0m
 * vertical spec), double-click = fullscreen. WebGL2 fallback is automatic. */
(function () {
    'use strict';

    const HEAD_BOX = { lateral: 0.5, up: 0.2, down: 1.0 };
    const canvas = document.getElementById('bubble');
    const overlay = document.getElementById('bubble-overlay');

    function fail(msg) {
        overlay.textContent = msg;
        overlay.classList.add('err');
    }

    if (!window.pc) { fail('viewer script failed to load'); return; }

    let app;
    try {
        app = new pc.Application(canvas, {
            mouse: new pc.Mouse(canvas),
            touch: new pc.TouchDevice(canvas),
            graphicsDeviceOptions: { antialias: false, alpha: false }
        });
    } catch (e) {
        fail('WebGL2 unavailable in this browser — use the SuperSplat button below');
        return;
    }

    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);

    function resize() {
        const r = canvas.parentElement.getBoundingClientRect();
        const h = document.fullscreenElement ? window.innerHeight
                                             : Math.round(r.width * 9 / 16);
        app.resizeCanvas(r.width, h);
    }
    window.addEventListener('resize', resize);
    document.addEventListener('fullscreenchange', resize);

    const camera = new pc.Entity('cam');
    camera.addComponent('camera', {
        clearColor: new pc.Color(0.05, 0.07, 0.09),
        fov: 75,
        nearClip: 0.05,
        farClip: 600
    });
    app.root.addChild(camera);

    // ---- look / move state -------------------------------------------------
    let yaw = 0, pitch = 0, fov = 75;
    const pos = new pc.Vec3(0, 0, 0);
    const keys = {};

    function applyCamera() {
        camera.setLocalPosition(pos);
        camera.setLocalEulerAngles(pitch, yaw, 0);
        camera.camera.fov = fov;
    }

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        const scale = 0.12 * (fov / 75);
        yaw -= (e.clientX - lastX) * scale;
        pitch -= (e.clientY - lastY) * scale;
        pitch = Math.max(-89, Math.min(89, pitch));
        lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        fov = Math.max(35, Math.min(100, fov + e.deltaY * 0.05));
    }, { passive: false });
    canvas.addEventListener('dblclick', () => {
        if (document.fullscreenElement) document.exitFullscreen();
        else canvas.parentElement.requestFullscreen();
    });
    window.addEventListener('keydown', (e) => { keys[e.key.toLowerCase()] = true; });
    window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

    app.on('update', (dt) => {
        const speed = 0.6 * dt;
        const fwd = camera.forward, right = camera.right;
        let dx = 0, dz = 0, dy = 0;
        if (keys.w || keys.arrowup) { dx += fwd.x; dz += fwd.z; }
        if (keys.s || keys.arrowdown) { dx -= fwd.x; dz -= fwd.z; }
        if (keys.d || keys.arrowright) { dx += right.x; dz += right.z; }
        if (keys.a || keys.arrowleft) { dx -= right.x; dz -= right.z; }
        if (keys.q) dy -= 1;
        if (keys.e) dy += 1;
        pos.x += dx * speed; pos.z += dz * speed; pos.y += dy * speed;
        // clamp to the pipeline's head-box: the region the gates certified
        pos.x = Math.max(-HEAD_BOX.lateral, Math.min(HEAD_BOX.lateral, pos.x));
        pos.z = Math.max(-HEAD_BOX.lateral, Math.min(HEAD_BOX.lateral, pos.z));
        pos.y = Math.max(-HEAD_BOX.down, Math.min(HEAD_BOX.up, pos.y));
        applyCamera();
    });

    // ---- splat asset -------------------------------------------------------
    const asset = new pc.Asset('machu', 'gsplat', { url: 'scene.sog' });
    asset.on('progress', (received, length) => {
        if (length) {
            overlay.textContent =
                'loading splat — ' + Math.round(100 * received / length) + '%';
        }
    });
    asset.on('load', () => {
        const splat = new pc.Entity('splat');
        splat.addComponent('gsplat', { asset: asset });
        app.root.addChild(splat);
        overlay.classList.add('hidden');
    });
    asset.on('error', (err) => fail('splat load failed: ' + err));
    app.assets.add(asset);
    app.assets.load(asset);

    applyCamera();
    resize();
    app.start();
})();
