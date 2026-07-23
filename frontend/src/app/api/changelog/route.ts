import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

export interface ChangelogEntry {
  text: string;
  version: string;
  date: string;
  link: string | null;
}

export async function GET() {
  try {
    const changelogPath = path.join(process.cwd(), '..', 'CHANGELOG.md');
    let content = '';
    
    try {
      content = await fs.readFile(changelogPath, 'utf-8');
    } catch {
      // Fallback if deployed and file is not at '..'
      return NextResponse.json({ entries: [] });
    }

    const lines = content.split('\n');
    const entries: ChangelogEntry[] = [];
    let currentVersion = '';
    let currentDate = '';
    let currentLink = '';

    for (const line of lines) {
      const versionMatch = line.match(/^##\s+\[(.*?)\](?:\s+-\s+(.*))?/);
      if (versionMatch) {
        currentVersion = versionMatch[1];
        currentDate = versionMatch[2] || '';
        currentLink = `https://github.com/Adit-Jain-srm/NightmareNet/releases/tag/v${currentVersion}`;
        continue;
      }
      
      if (currentVersion === 'Unreleased') {
        continue;
      }

      const bulletMatch = line.match(/^-\s+(.*)/);
      if (bulletMatch && currentVersion) {
        const text = bulletMatch[1];
        let link = currentLink;
        
        const prMatch = text.match(/\(#(\d+)\)/);
        if (prMatch) {
          link = `https://github.com/Adit-Jain-srm/NightmareNet/pull/${prMatch[1]}`;
        }
        
        entries.push({
          text,
          version: currentVersion,
          date: currentDate,
          link,
        });
        
        if (entries.length >= 5) {
          break;
        }
      }
    }
    
    return NextResponse.json({ entries });
  } catch (error) {
    console.error('Failed to parse changelog:', error);
    return NextResponse.json({ entries: [] }, { status: 500 });
  }
}
