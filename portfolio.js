'use strict';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const prefersReducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
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
      if (animate && !prefersReducedMotion) {
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

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      $$('.header-inner nav a').forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`);
      });
    });
  }, { rootMargin: '-15% 0px -65% 0px' });
  ['about', 'skills', 'projects', 'experience', 'contact'].forEach(id => observer.observe($(`#${id}`)));
}

$('#year').textContent = new Date().getFullYear();
filterProjects();
