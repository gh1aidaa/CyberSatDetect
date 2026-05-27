// خلفية فضاء: نجوم فقط — بدون قبة/كورة
let renderer, scene, camera, stars, controls;

(function init(){
  const container = document.getElementById("canvas_container");
  if (!container) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 5000);
  camera.position.set(0, 0, 300);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // شفاف — يبين لون الباكقراوند من CSS
  container.appendChild(renderer.domElement);

  // دوران خفيف للكاميرا (ممكن توقيفه)
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.enablePan = false;
  controls.enableZoom = false;

  // نجوم (Particles) فقط
  const starCount = 1200; // كثافة النجوم
  const positions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const radius = 1800 * Math.cbrt(Math.random()); // توزيع كروي واسع
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos((Math.random() * 2) - 1);
    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);
    positions.set([x,y,z], i*3);
  }

  const starsGeo = new THREE.BufferGeometry();
  starsGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const starsMat = new THREE.PointsMaterial({
    size: 2,
    color: 0xFFFFFF,
    transparent: true,
    opacity: 0.9,
    depthWrite: false
  });
  stars = new THREE.Points(starsGeo, starsMat);
  scene.add(stars);

  window.addEventListener("resize", onResize);
  animate();
})();

function animate(){
  requestAnimationFrame(animate);
  // دوران بسيط للنجوم
  stars.rotation.y += 0.0006;
  if (controls) controls.update();
  renderer.render(scene, camera);
}

function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
