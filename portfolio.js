'use strict';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const reducedMotionPreference = matchMedia('(prefers-reduced-motion: reduce)');
let activeCategory = 'All';

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === 'dark';
  $('.theme-toggle').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  try { localStorage.setItem('tarun-readable-theme', theme); } catch {}
}

let initialTheme = 'light';
try { initialTheme = localStorage.getItem('tarun-readable-theme') || 'light'; } catch {}
applyTheme(initialTheme === 'dark' ? 'dark' : 'light');
$('.theme-toggle').addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

function closeNavigation() {
  $('#mobile-nav').hidden = true;
  $('.menu-toggle').setAttribute('aria-expanded', 'false');
}

$('.menu-toggle').addEventListener('click', () => {
  const open = $('#mobile-nav').hidden;
  $('#mobile-nav').hidden = !open;
  $('.menu-toggle').setAttribute('aria-expanded', String(open));
});
$$('#mobile-nav a').forEach(link => link.addEventListener('click', closeNavigation));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('#mobile-nav').hidden) {
    closeNavigation();
    $('.menu-toggle').focus();
  }
});

function filterProjects(animate = false) {
  const query = $('#project-search').value.trim().toLowerCase();
  let count = 0;
  $$('.project-card').forEach(card => {
    const matches = (activeCategory === 'All' || card.dataset.category === activeCategory)
      && card.textContent.toLowerCase().includes(query);
    card.hidden = !matches;
    card.classList.remove('filter-enter');
    if (matches) {
      count++;
      if (animate && !reducedMotionPreference.matches) {
        requestAnimationFrame(() => card.classList.add('filter-enter'));
      }
    }
  });
  $('.project-count').textContent = count === 17 ? 'Showing all 17 projects' : `Showing ${count} of 17 projects`;
  $('.empty-state').hidden = count !== 0;
  $$('[data-filter]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.filter === activeCategory));
  });
}

$$('[data-filter]').forEach(button => button.addEventListener('click', () => {
  activeCategory = button.dataset.filter;
  filterProjects(true);
}));
$('#project-search').addEventListener('input', () => filterProjects());
$('#reset-filters').addEventListener('click', () => {
  activeCategory = 'All';
  $('#project-search').value = '';
  filterProjects(true);
});

// Project links always reveal their destination, even when a filter is active.
$$('a[href^="#project-"]').forEach(link => link.addEventListener('click', () => {
  activeCategory = 'All';
  $('#project-search').value = '';
  filterProjects();
}));

$('#year').textContent = new Date().getFullYear();
filterProjects();

// Motion enhances visible content; nothing depends on an animation to be readable.
const motionAnimations = new Set();
function animateEntrance(element, delay = 0, distance = 24) {
  if (reducedMotionPreference.matches || !element.animate) return;
  const animation = element.animate([
    { opacity: .55, translate: `0 ${distance}px` },
    { opacity: 1, translate: '0 0' }
  ], { duration: 620, delay, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'backwards' });
  motionAnimations.add(animation);
  const forget = () => motionAnimations.delete(animation);
  animation.addEventListener('finish', forget, { once: true });
  animation.addEventListener('cancel', forget, { once: true });
}

if ('IntersectionObserver' in window) {
  const entrances = new IntersectionObserver(entries => {
    let stagger = 0;
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entrances.unobserve(entry.target);
      // Skip elements already in view on load so the introduction is immediate.
      if (entry.target.getBoundingClientRect().top > 0) {
        animateEntrance(entry.target, Math.min(stagger++ * 65, 195));
      }
    });
  }, { threshold: .08 });
  $$('.section-heading, .section-label, .about-content, .skills-grid article, .project-card, .experience-card, .awards-grid article, .education-grid article, .certifications, .contact-layout').forEach(element => {
    if (element.getBoundingClientRect().top >= innerHeight) entrances.observe(element);
  });
}

$$('.project-card details').forEach(details => details.addEventListener('toggle', () => {
  if (details.open) animateEntrance(details.querySelector('ul'), 0, 8);
}));

const progressLine = $('.reading-progress');
const backToTop = $('.back-to-top');
const hero = $('.hero');
let scrollFrame = 0;
function updateScrollEffects() {
  scrollFrame = 0;
  const range = document.documentElement.scrollHeight - innerHeight;
  const progress = range > 0 ? Math.min(1, Math.max(0, scrollY / range)) : 0;
  progressLine.style.transform = `scaleX(${progress})`;
  const header = $('.site-header');
  header.classList.toggle('is-scrolled', scrollY > 24);
  let currentSection = '';
  ['about', 'skills', 'projects', 'experience', 'contact'].forEach(id => {
    if ($(`#${id}`).getBoundingClientRect().top <= header.offsetHeight + 100) currentSection = id;
  });
  $$('.header-inner nav a').forEach(link => {
    const active = link.getAttribute('href') === `#${currentSection}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  // Leave a focused control available until focus moves away.
  backToTop.hidden = scrollY < innerHeight && document.activeElement !== backToTop;
  hero.style.setProperty('--hero-drift', reducedMotionPreference.matches ? '0px' : `${Math.min(scrollY * .12, 60)}px`);
}
function scheduleScrollEffects() {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScrollEffects);
}
addEventListener('scroll', scheduleScrollEffects, { passive: true });
addEventListener('resize', scheduleScrollEffects);
backToTop.addEventListener('blur', scheduleScrollEffects);
backToTop.addEventListener('click', event => {
  event.preventDefault();
  // Focus the home link before hiding this floating control.
  $('.brand').focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: reducedMotionPreference.matches ? 'instant' : 'smooth' });
});
if ('ResizeObserver' in window) {
  new ResizeObserver(scheduleScrollEffects).observe(document.body);
}
reducedMotionPreference.addEventListener('change', () => {
  if (reducedMotionPreference.matches) motionAnimations.forEach(animation => animation.cancel());
  scheduleScrollEffects();
});
updateScrollEffects();
