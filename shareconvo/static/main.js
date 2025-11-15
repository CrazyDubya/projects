document.addEventListener('DOMContentLoaded', () => {
  // --- VOTE HANDLING ---
  const voteButtons = document.querySelectorAll('.vote-btn');
  voteButtons.forEach(button => {
    button.addEventListener('click', async (event) => {
      const convoId = event.target.closest('li').dataset.convoId;
      const voteType = event.target.dataset.voteType;

      const response = await fetch(`/vote/${convoId}/${voteType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        const convoElem = document.querySelector(`li[data-convo-id='${convoId}']`);
        convoElem.querySelector('.likes-count').textContent = data.likes;
        convoElem.querySelector('.dislikes-count').textContent = data.dislikes;
      } else {
        console.error('Vote failed');
      }
    });
  });

  // --- FEEDBACK PROMPT HANDLING ---
  const feedbackBanner = document.getElementById('feedback-banner');
  const feedbackConvoTitle = document.getElementById('feedback-convo-title');
  const closeFeedbackBannerBtn = document.getElementById('close-feedback-banner');

  // 1. Log clicks on "View Full Conversation"
  const viewLinks = document.querySelectorAll('.view-convo-link');
  viewLinks.forEach(link => {
    link.addEventListener('click', (event) => {
      const convoElem = event.target.closest('li');
      const convoId = convoElem.dataset.convoId;
      const convoTitle = convoElem.querySelector('h2').textContent;
      // Store the convo info for when the user returns
      sessionStorage.setItem('lastViewedConvoId', convoId);
      sessionStorage.setItem('lastViewedConvoTitle', convoTitle);
    });
  });

  // 2. Use Page Visibility API to show prompt
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      const convoId = sessionStorage.getItem('lastViewedConvoId');
      const convoTitle = sessionStorage.getItem('lastViewedConvoTitle');

      if (convoId && convoTitle) {
        // Check if feedback has already been given for this convo in this session
        const feedbackGiven = sessionStorage.getItem(`feedbackGivenFor_${convoId}`);
        if (!feedbackGiven) {
          feedbackConvoTitle.textContent = `'${convoTitle}'`;
          feedbackBanner.classList.remove('hidden');
        }
      }
    }
  });

  // 3. Allow user to close the banner
  closeFeedbackBannerBtn.addEventListener('click', () => {
    feedbackBanner.classList.add('hidden');
    // Also mark it as "feedback given" for this session to not show again
    const convoId = sessionStorage.getItem('lastViewedConvoId');
    if (convoId) {
      sessionStorage.setItem(`feedbackGivenFor_${convoId}`, 'true');
    }
  });

  // --- FEEDBACK FORM UI HANDLING ---
  const feedbackForm = document.getElementById('feedback-form');
  const accuracyButtons = document.querySelectorAll('.accuracy-btn');
  const starsContainer = document.querySelector('.stars');
  const reportBtn = document.getElementById('report-btn');
  const feedbackAccuracyInput = document.getElementById('feedback-accuracy');
  const feedbackRatingInput = document.getElementById('feedback-rating');
  const feedbackReportInput = document.getElementById('feedback-report');

  accuracyButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      accuracyButtons.forEach(btn => btn.classList.remove('active'));
      e.currentTarget.classList.add('active');
      feedbackAccuracyInput.value = e.currentTarget.dataset.accuracy;
    });
  });

  starsContainer.addEventListener('click', (e) => {
    if (e.target.classList.contains('star')) {
      const rating = e.target.dataset.value;
      starsContainer.dataset.rating = rating;
      feedbackRatingInput.value = rating;
    }
  });

  reportBtn.addEventListener('click', (e) => {
    e.currentTarget.classList.toggle('active');
    feedbackReportInput.value = e.currentTarget.classList.contains('active') ? '1' : '0';
  });

  // --- FEEDBACK FORM SUBMISSION ---
  feedbackForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const convoId = sessionStorage.getItem('lastViewedConvoId');
    if (!convoId) {
      console.error('No conversation ID found for feedback.');
      return;
    }

    const formData = {
      accuracy: feedbackAccuracyInput.value,
      rating: feedbackRatingInput.value,
      report: feedbackReportInput.value,
    };

    // Basic validation
    if (!formData.accuracy || !formData.rating) {
      alert('Please provide both accuracy and a star rating.');
      return;
    }

    const response = await fetch(`/feedback/${convoId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });

    if (response.ok) {
      feedbackBanner.classList.add('hidden');
      sessionStorage.setItem(`feedbackGivenFor_${convoId}`, 'true');
      alert('Thank you for your feedback!');
    } else {
      console.error('Failed to submit feedback.');
      alert('There was an error submitting your feedback.');
    }
  });
});
