# WCAG 2.1 AA Accessibility Audit

## Scope

- Landing page
- Dashboard and all dashboard panels
- Keyboard navigation
- Focus visibility and order
- Screen-reader announcements
- Text and UI contrast

## Automated checks

Command:

```bash
npm run test:a11y
```

Environment:

- Browser:
- Operating system:
- Commit:
- Date:

### Before remediation

| Route | Critical | Serious | Moderate | Minor |
|---|---:|---:|---:|---:|
| `/` |  |  |  |  |
| `/dashboard` |  |  |  |  |

### After remediation

| Route | Critical | Serious | Moderate | Minor |
|---|---:|---:|---:|---:|
| `/` | 0 | 0 | 0 | 0 |
| `/dashboard` | 0 | 0 | 0 | 0 |

## Manual keyboard audit

- [ ] Skip link is the first focusable element.
- [ ] Tab order follows the visual and reading order.
- [ ] Every interactive control is reachable.
- [ ] Enter activates links and buttons.
- [ ] Space activates buttons and checkbox-like controls.
- [ ] Escape closes dialogs, menus, and popovers.
- [ ] Focus returns to the control that opened a dialog/menu.
- [ ] No keyboard traps are present.
- [ ] Focus indicators remain visible on dark and light surfaces.

## Screen-reader audit

Screen reader used:

- [ ] Page title and primary heading are announced.
- [ ] Landmarks are available: header, nav, main, footer.
- [ ] Icon-only controls have accessible names.
- [ ] Form controls have programmatic labels.
- [ ] Validation errors are associated with their fields.
- [ ] Loading, success, and error changes use live regions.
- [ ] Decorative graphics are hidden from assistive technology.
- [ ] Charts have text alternatives or summaries.

## Contrast audit

- [ ] Normal text meets 4.5:1.
- [ ] Large text meets 3:1.
- [ ] UI components and focus indicators meet 3:1.
- [ ] Status is not communicated by color alone.

## Components remediated

| Component | Semantic controls | Keyboard | Accessible names | Focus | Live region | Contrast |
|---|---|---|---|---|---|---|
| Navbar |  |  |  |  | N/A |  |
| Hero |  |  |  |  |  |  |
| Playground |  |  |  |  |  |  |
| TrainingLab |  |  |  |  |  |  |
| Dashboard shell |  |  |  |  |  |  |
