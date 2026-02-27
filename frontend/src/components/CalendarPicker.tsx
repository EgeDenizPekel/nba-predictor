import { useState } from 'react';

const MIN = { year: 2003, month: 9, day: 28 }; // Oct 28 2003 (month is 0-indexed)
const MAX = { year: 2022, month: 3, day: 10 };  // Apr 10 2022

const YEARS = Array.from({ length: MAX.year - MIN.year + 1 }, (_, i) => MIN.year + i);
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];
const DOW = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

function ymd(year: number, month: number, day: number): string {
  return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
}

function isBeforeMin(y: number, m: number, d: number): boolean {
  if (y !== MIN.year) return y < MIN.year;
  if (m !== MIN.month) return m < MIN.month;
  return d < MIN.day;
}

function isAfterMax(y: number, m: number, d: number): boolean {
  if (y !== MAX.year) return y > MAX.year;
  if (m !== MAX.month) return m > MAX.month;
  return d > MAX.day;
}

interface Props {
  value: string; // YYYY-MM-DD
  onChange: (date: string) => void;
}

export default function CalendarPicker({ value, onChange }: Props) {
  // Parse selected date in local time
  const sel = value
    ? { year: Number(value.slice(0, 4)), month: Number(value.slice(5, 7)) - 1, day: Number(value.slice(8, 10)) }
    : null;

  const [viewYear, setViewYear] = useState(sel?.year ?? 2021);
  const [viewMonth, setViewMonth] = useState(sel?.month ?? 10); // November 2021

  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
  const startDow = new Date(viewYear, viewMonth, 1).getDay();

  const prevMonth = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(y => y - 1); }
    else setViewMonth(m => m - 1);
  };

  const nextMonth = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(y => y + 1); }
    else setViewMonth(m => m + 1);
  };

  const handleYearChange = (year: number) => {
    setViewYear(year);
    // Clamp month to valid range if at boundaries
    if (year === MIN.year && viewMonth < MIN.month) setViewMonth(MIN.month);
    if (year === MAX.year && viewMonth > MAX.month) setViewMonth(MAX.month);
  };

  const handleMonthChange = (month: number) => {
    setViewMonth(month);
  };

  // Valid months for the current year (clamped to dataset boundaries)
  const validMonths = MONTH_NAMES.map((name, i) => ({ name, index: i })).filter(({ index }) => {
    if (viewYear === MIN.year && index < MIN.month) return false;
    if (viewYear === MAX.year && index > MAX.month) return false;
    return true;
  });

  const isPrevDisabled = viewYear === MIN.year && viewMonth === MIN.month;
  const isNextDisabled = viewYear === MAX.year && viewMonth === MAX.month;

  return (
    <div className="w-full select-none">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={prevMonth}
          disabled={isPrevDisabled}
          className="w-7 h-7 flex items-center justify-center rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed text-gray-600 font-medium"
        >
          ‹
        </button>

        <div className="flex items-center gap-1">
          <select
            value={viewMonth}
            onChange={(e) => handleMonthChange(Number(e.target.value))}
            className="text-sm font-medium text-gray-700 bg-transparent border border-gray-200 rounded px-1 py-0.5 cursor-pointer focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            {validMonths.map(({ name, index }) => (
              <option key={index} value={index}>{name}</option>
            ))}
          </select>
          <select
            value={viewYear}
            onChange={(e) => handleYearChange(Number(e.target.value))}
            className="text-sm font-semibold text-gray-900 bg-transparent border border-gray-200 rounded px-1 py-0.5 cursor-pointer focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        <button
          onClick={nextMonth}
          disabled={isNextDisabled}
          className="w-7 h-7 flex items-center justify-center rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed text-gray-600 font-medium"
        >
          ›
        </button>
      </div>

      {/* Day-of-week headers */}
      <div className="grid grid-cols-7 mb-1">
        {DOW.map((d) => (
          <div key={d} className="text-center text-xs text-gray-400 py-0.5">{d}</div>
        ))}
      </div>

      {/* Day grid */}
      <div className="grid grid-cols-7 gap-y-0.5">
        {/* Empty cells before first day */}
        {Array.from({ length: startDow }).map((_, i) => (
          <div key={`e-${i}`} />
        ))}

        {Array.from({ length: daysInMonth }, (_, i) => i + 1).map((day) => {
          const disabled = isBeforeMin(viewYear, viewMonth, day) || isAfterMax(viewYear, viewMonth, day);
          const isSelected = sel?.year === viewYear && sel?.month === viewMonth && sel?.day === day;

          return (
            <button
              key={day}
              disabled={disabled}
              onClick={() => onChange(ymd(viewYear, viewMonth, day))}
              className={[
                'mx-auto w-7 h-7 flex items-center justify-center rounded-full text-xs transition-colors',
                isSelected
                  ? 'bg-blue-600 text-white font-semibold'
                  : disabled
                  ? 'text-gray-300 cursor-not-allowed'
                  : 'text-gray-700 hover:bg-blue-100 hover:text-blue-700 cursor-pointer',
              ].join(' ')}
            >
              {day}
            </button>
          );
        })}
      </div>
    </div>
  );
}
