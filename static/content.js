function content(e) {
  localStorage.setItem('title', e.getAttribute('title'));
  window.document.location="recommend?book="+e.getAttribute('title');
}
